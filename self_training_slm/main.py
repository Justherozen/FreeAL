from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer, HfArgumentParser, set_seed
from transformers.trainer import *
from modeling import MODEL, AutoTokenizer
from datasets import ClassificationDataset,MyDataCollator
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from utils.utils import set_logger, path_checker, metrics_fn
from runner import Runner
import torch
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCEWithLogitsLoss
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from mytrainer import MyTrainer
from typing import List, Dict

def main():
    # Get arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.num_labels = data_args.num_labels

    # Path check and set logger
    path_checker(training_args)
    set_logger(training_args)

    # Get model name
    model_name = model_args.model_name_or_path \
        if model_args.model_name_or_path is not None \
        else MODEL[model_args.model.lower()] \
        if model_args.model.lower() in MODEL \
        else model_args.model

    # Set seed
    set_seed(training_args.seed)

    # Set model
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_name, cache_dir=model_args.cache_dir, num_labels=data_args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_name, cache_dir=model_args.cache_dir)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, cache_dir=model_args.cache_dir)

    train_aug = ClassificationDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length,
                                  data_args.overwrite_cache, mode="train_chat") if training_args.do_train else None
    train = ClassificationDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length,
                                  data_args.overwrite_cache, mode="train") if training_args.do_train else None
    dev = ClassificationDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length,
                                data_args.overwrite_cache, mode="dev") if training_args.do_eval else None
    test = ClassificationDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length,
                                 data_args.overwrite_cache, mode="test") if training_args.do_predict else None
    
    if data_args.learning_setting == 'transductive':
        #transductive setting
        eval_dataset = train
    else:
        #inductive setting
        eval_dataset = test
    training_args.learning_setting = data_args.learning_setting
        
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_aug,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_fn,
        data_collator=MyDataCollator()
    )

    # Set runner
    runner = Runner(
        model_name=model_name,
        trainer=trainer,
        tokenizer=tokenizer,
        training_args=training_args,
        test=test,
        eval=dev)

    # Start
    runner()


if __name__ == "__main__":
    main()
