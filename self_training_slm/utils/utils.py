import logging
import os
import numpy as np

def set_logger(training_args):

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.FileHandler(training_args.logging_dir + "/logging.log", 'w', encoding='utf-8'),
                  logging.StreamHandler()]
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


def path_checker(training_args):
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if not os.path.exists(training_args.logging_dir):
        os.mkdir(training_args.logging_dir)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def metrics_fn(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


def prediction(logit):
    return np.argmax(logit, axis=1)