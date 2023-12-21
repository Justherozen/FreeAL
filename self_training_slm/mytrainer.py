from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer, HfArgumentParser, set_seed
from transformers.trainer import *
from modeling import MODEL, AutoTokenizer
from datasets import ClassificationDataset
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from utils.utils import set_logger, path_checker, metrics_fn
from runner import Runner
import torch
from torch import nn
import os
import copy
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCEWithLogitsLoss
from sklearn.metrics import confusion_matrix ,recall_score,roc_auc_score, accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


class MyTrainer(Trainer):
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )

                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            
            ##------------------------------------------------------------------
        
            ##Start of sample selection and demonstration pool construction
            self.temp_u = args.temp_u
            length = len(train_dataloader.dataset)
            chosen_list = torch.zeros((length)).cuda()
            chosen_list_sel = torch.zeros((length)).cuda()
            self.num_labels = args.num_labels
            if epoch > 0:
                t_model = copy.deepcopy(model).to(args.device)
                t_model.eval()
                rho_sel = 0.2
                
                num_labels = args.num_labels
                targets_all = torch.zeros((length),dtype=torch.long).cuda()
                outputs_all = torch.zeros((length,num_labels)).cuda()
                loss_all = torch.zeros((length)).cuda()
                embeddings_all = torch.zeros((length,args.embedding_dim))
                new_dataloader = train_dataloader
                # Calculate the losses for each training sample
                for step, inputs in enumerate(new_dataloader):
                    inputs = self._prepare_inputs(inputs)
                    with torch.no_grad():
                        labels = inputs['labels']
                        index = inputs['index']
                        valid_mask = (labels >= 0)
                        del inputs['index']
                        del inputs['labels']
                        myinputs = {
                            "input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"],
                        }
                        output = t_model(**myinputs,output_hidden_states=True)
                        logits1 = output['logits']
                        embeddings = output['hidden_states'][-1][:,0,:]
                        embeddings_all[index] = embeddings.detach().cpu()
                        outputs_all[index] = logits1
                        targets_all[index] = labels
                        loss_fct = nn.CrossEntropyLoss(reduction='none')
                        loss = loss_fct(logits1[valid_mask], labels.view(-1)[valid_mask])
                        loss_all[index[valid_mask]] = loss
        
                high_conf_all = outputs_all.max(dim=-1)[0]
                pred_idx_all = outputs_all.max(dim=-1)[1]
                pred_label_all = pred_idx_all.cpu().tolist()
                #filter out the invalid sample with label==-1 (ambiguous annotation by LLM)
                valid_idx_all = targets_all >= 0 
                matched_idx_all = (pred_idx_all==(targets_all))&valid_idx_all             
                
                #GMM selection for robust self-training of SLM
                loss_all = ((loss_all-loss_all[valid_idx_all].min())/(loss_all[valid_idx_all].max()-loss_all[valid_idx_all].min())).detach().cpu().reshape(-1,1)
                loss_all_tmp = loss_all[torch.where(valid_idx_all)[0].cpu()]
                gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
                gmm.fit(loss_all_tmp)
                prob = gmm.predict_proba(loss_all_tmp) 
                prob = prob[:,gmm.means_.argmin()]
                chosen_idx_all_gmm = np.where(prob > 0.7)[0]
                chosen_list[torch.where(valid_idx_all)[0][chosen_idx_all_gmm]] = 1
                
                #Class-wise demonstration pool construction as feedback to LLM
                chosen_top_indices = []
                for j in range(num_labels):
                    index_j_matched = torch.where((pred_idx_all==j)&matched_idx_all)[0].cpu()
                    max_score_j = high_conf_all[index_j_matched]
                    if index_j_matched.shape[0]==0:
                        continue
                    sort_index_j = (-max_score_j).sort()[1].cpu().numpy()
                    partition_j_sel = int((index_j_matched.shape[0])*rho_sel)
                    if (partition_j_sel) == 0:
                        continue
                    index_j_sel = index_j_matched[sort_index_j[:partition_j_sel]]
                    chosen_list_sel[index_j_sel] = 1
                    
                    # For these clean samples [index_j_sel], adopt k-medoids clustering
                    embeddings_j = embeddings_all[index_j_sel]
                    # k-medoids for representative samples, 100/2=50 medoids samples
                    num_clusters = args.select_demo_num//num_labels
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_j)
                    kmeans_labels = kmeans.labels_
                    idx_all_representative = []
                    embedding_all_representative = []
                    for k in range(num_clusters):
                        vectors_in_cluster = embeddings_j[kmeans_labels==k]
                        idx_in_cluster = (index_j_sel)[kmeans_labels==k]
                        centroid = vectors_in_cluster.mean(dim=0)
                        distances_to_centroid = torch.norm(vectors_in_cluster - centroid, dim=1)
                        index_of_representative = torch.argmin(distances_to_centroid)
                        embedding_all_representative.append(vectors_in_cluster[index_of_representative])
                        idx_all_representative.append(idx_in_cluster[index_of_representative].reshape(1))
                    
                    #representation and index of samples to be demonstrations
                    embedding_all_representative = torch.cat([emb.reshape(1,-1) for emb in embedding_all_representative])
                    idx_all_representative = torch.cat(idx_all_representative)
                
                    #demonstration retrieval for LLMs: similarity between each training example and demonstration sample
                    cos_similarities = cosine_similarity(embeddings_all, embedding_all_representative)
                    sort_result = torch.sort(torch.from_numpy(cos_similarities), dim=1, descending=True)
                    top_indices = sort_result[1][:, :(args.shot_num//num_labels)]
                    #similarity-based demonstration retrieval
                    for i in range(top_indices.shape[0]):
                        top_indices[i,:] = idx_all_representative[top_indices[i,:]]
                    chosen_top_indices.append(top_indices)
                
                chosen_top_indices = torch.cat(chosen_top_indices,dim=1)
                chosen_idx_sel = torch.where(chosen_list_sel)[0].cpu()
                if args.learning_setting=='transductive' and epoch == 1:    
                    import pickle
                    # saving the feedbacks 
                    # clean sample index
                    with open("./feedback/right_list_mr.pkl","wb") as f:
                        pickle.dump(chosen_idx_sel.tolist(),f)
                    # demonstration retrieval
                    with open("./feedback/demo_index_mr.pkl","wb") as f:
                        pickle.dump(chosen_top_indices.tolist(),f)
                    # pseudo-labels for all samples
                    with open("./feedback/pred_label_mr.pkl","wb") as f:
                        pickle.dump(pred_label_all,f)

            if epoch > args.warmup:
                self.chosen_list = chosen_list.to(args.device)
            else:
                #warm-up phase
                self.chosen_list = torch.ones_like(chosen_list).to(args.device)
            
            #end of the sample selection and demonstration construction
            
            
            
            
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    (total_batched_samples % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        w = linear_rampup(self.state.epoch,10)
        is_in_train = ("input_ids_bt" in inputs.keys())
        
        labels = inputs.get("labels")
        if is_in_train:
            valid_mask = (labels >= 0)
            batch_index = inputs['index']
            batch_mask = self.chosen_list[batch_index]
            batch_mask = batch_mask.bool()&valid_mask
            batch_mask_un = ~(self.chosen_list[batch_index].bool())&valid_mask
        else:
            batch_mask = torch.ones_like(labels)
            batch_mask_un = torch.zeros_like(labels)
        index_x = torch.where(batch_mask)[0]
        index_u = torch.where(batch_mask_un)[0]

        myinputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        embeddings = model.roberta(**myinputs)[0] 
        logits = model.classifier(embeddings) 
        
        loss_fct = nn.CrossEntropyLoss(reduction='none') 
        loss_fct_un = nn.KLDivLoss(reduction='sum')
        
        #cross-entropy loss on filtered clean subset
        loss_l = loss_fct(logits[index_x], labels.view(-1)[index_x]).mean()
        
        #embedding-level mix-up stragety on clean subset
        valid_embeddings = embeddings[index_x]
        with torch.no_grad():
            targets_l = F.one_hot(labels[index_x].detach(),num_classes=self.num_labels)
            all_targets = targets_l.detach()
        rand_idx = torch.randperm(valid_embeddings.size(0))
        l = np.random.beta(4,4) 
        l = max(l, 1 - l)
        mixed_embeddings = l * valid_embeddings + (1 - l) * valid_embeddings[rand_idx]
        mixed_targets = l * all_targets + (1 - l) * all_targets[rand_idx]
        mixed_logits = model.classifier(mixed_embeddings)
        loss_mix = -torch.mean(torch.sum(F.log_softmax(mixed_logits, dim=-1) * mixed_targets, dim=-1)) 
        final_loss = (loss_l) + w*loss_mix
        
        #in training: utilize consistency regularization on all data
        if is_in_train:
            myinputs_bt = {
                "input_ids": inputs["input_ids_bt"],
                "attention_mask": inputs["attention_mask_bt"],
            }
            embeddings_bt = model.roberta(**myinputs_bt)[0] 
            logits_bt = model.classifier(embeddings_bt) 
            loss_cr = loss_fct(logits_bt[index_x], labels.view(-1)[index_x]).mean()
            loss_cr_un = 0.5 * loss_fct_un(F.log_softmax(logits[index_u]/self.temp_u,dim=-1), F.softmax(logits_bt[index_u].detach().clone(),dim=-1)) + 0.5 * loss_fct_un(F.log_softmax(logits_bt[index_u]/self.temp_u,dim=-1), F.softmax(logits[index_u].detach().clone(),dim=-1))
            final_loss = final_loss + w *(loss_cr + loss_cr_un)
        
        return (final_loss, {'outputs':logits}) if return_outputs else final_loss
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
       
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps       
        
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()