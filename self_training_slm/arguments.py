from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments as OriginalTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="BERT",
        metadata={"help": "Model name (BERT, BART, ALBERT, ... )"}
    )

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=".cache", metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    
    hidden_dropout_prob: Optional[float] = field(
        default=0, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(default="classification", metadata={"help": "The name of the task"})
    data_dir: str = field(default="data_in", metadata={"help": "Should contain the data files for the task."})
    num_labels: int = field(default=2, metadata={"help": "The number of labels on dataset"})
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    learning_setting: str = field(
        default="transductive", metadata={"help": "Adopt the inductive learning or transductive learning setting"}
    )


@dataclass
class TrainingArguments(OriginalTrainingArguments):
    
    output_dir: str = field(
        default="data_out",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    # num_labels: int = field(default=2, metadata={"help": "The number of labels on dataset"})
    logging_dir: Optional[str] = field(default="data_out", metadata={"help": "Tensorboard log dir."})
    embedding_dim: int = field(default=768, metadata={"help": "The dim of last-layer embedding of the backbone, default 768 for RoBERTa-base"})
    warmup: int = field(default=3, metadata={"help": "The number of warm-up epochs"})
    temp_u: float = field(default=0.85, metadata={"help": "The temperature for unlabeled data, following UDA"})
    select_demo_num: int = field(default=100, metadata={"help": "The number of selected demonsration samples to feedback LLMs"})
    shot_num: int = field(default=10, metadata={"help": "The K-number of few-shot demonsration retrieval"})