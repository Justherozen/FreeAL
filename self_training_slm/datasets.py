import os
import torch
import tqdm
import logging
import csv
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Optional,Dict
from dataclasses import dataclass
from filelock import FileLock
from transformers import pipeline

logger = logging.getLogger(__name__)


class MyDataCollator:
    def __init__(self):
        pass
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0],dict):
            #with data-augmentation through backtranslation
            inputs = [example["inputs"] for example in examples]
            inputs_bt = [example["inputs_bt"] for example in examples]
            index = [ip.index for ip in inputs]
            input_ids = [ip.input_ids for ip in inputs]
            attention_mask = [ip.attention_mask for ip in inputs]
            label = [ip.label for ip in inputs]
            input_ids_bt = [ip_bt.input_ids for ip_bt in inputs_bt]
            attention_mask_bt = [ip_bt.attention_mask for ip_bt in inputs_bt]
            batch = {
                'index': torch.tensor(index),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'labels': torch.tensor(label),
                'input_ids_bt': torch.tensor(input_ids_bt),
                'attention_mask_bt': torch.tensor(attention_mask_bt),
            }
            
        else:
            #without data augmentation
            index = [example.index for example in examples]
            input_ids = [example.input_ids for example in examples]
            attention_mask = [example.attention_mask for example in examples]
            label = [example.label for example in examples]
            batch = {
                'index': torch.tensor(index),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'labels': torch.tensor(label),
            }
        return batch

@dataclass
class DatasetInputExample:
    
    contents: str
    contents_bt: str
    label: Optional[int]
        

@dataclass
class DatasetInputFeature:
    
    index: int
    input_ids: List[int]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
            
            
class ClassificationDataset(Dataset):

    features: List[DatasetInputFeature]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: str = "train",
    ):

        processor = ClassificationProcessor()
        self.mode = mode
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length), task,),
        )
        
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                data_dir = os.path.join(data_dir,task)
                if mode == "dev":
                    examples = processor.get_dev_examples(data_dir)
                elif mode == "test":
                    examples = processor.get_test_examples(data_dir)
                elif mode == "train_chat":
                    examples = processor.get_train_chat_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)

                logger.info("Training examples: {}".format(len(examples)))
                self.features = convert_examples_to_features(examples, max_seq_length, tokenizer,mode)
                logger.info("Saving features into cached file {}".format(cached_features_file))
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        if self.mode == "train_chat":
            return self.features[i]
        else:
            return self.features[i]
    
class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_train_chat_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()
    
    
class ClassificationProcessor(DataProcessor):
    
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        train = self._read_csv(data_dir, "train")

        return self._create_examples(train,"train")
    
    def get_train_chat_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train chat".format(data_dir))
        train_chat = self._read_csv(data_dir, "train_chat_bt")
        
        return self._create_examples(train_chat,"train_chat")

    
    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        dev = self._read_csv(data_dir, "dev")
        
        return self._create_examples(dev,"dev")

    
    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        test = self._read_csv(data_dir, "test")
        
        return self._create_examples(test,"test")

    
    def _read_csv(self, input_dir, set_type):
        corpus = []
        with open("{}/{}.csv".format(input_dir, set_type), 'r', encoding='utf-8') as f:
            data = csv.reader(f)
            if set_type == "train_chat_bt":
                for line in data:
                    if len(line) == 1:
                        corpus.append([line[0]])
                    elif len(line) == 2:
                        corpus.append([line[1], int(line[0])])
                    else:
                        corpus.append([line[1], int(line[0]), line[2]])
            else:
                for line in data:
                    if len(line) == 1:
                        corpus.append([line[0]])
                    else:
                        corpus.append([line[1], int(line[0])])

        return corpus

    
    def _create_examples(self, corpus, mode):
        """Creates examples for the training and dev sets."""
        examples = []
        for data in tqdm.tqdm(corpus, desc="creating examples for "+mode):
    
            if mode == "train_chat":
                contents_bt = data[2]#pipe_bt(pipe(data[0])[0]["translation_text"])[0]["translation_text"]
            else:
                contents_bt = None
            examples.append(
                DatasetInputExample(
                    contents=data[0],
                    contents_bt = contents_bt,
                    label=data[1] if len(data) == 2 or len(data) == 3 else None
                )
            )
        return examples

        
def convert_examples_to_features(
    examples: List[DatasetInputExample], max_length: int, tokenizer: PreTrainedTokenizer, mode):
    """
    Loads a data file into a list of `DatasetInputExample`
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("\nWriting example {} of {}".format(ex_index, len(examples)))

        inputs = tokenizer(
            example.contents,
            max_length=max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True
        )
        

        if mode == "train_chat":
            inputs_bt = tokenizer(
                example.contents_bt,
                max_length=max_length,
                add_special_tokens=True,
                padding="max_length",
                truncation=True
            )
            features.append({
                "inputs":
                DatasetInputFeature(
                    index = ex_index,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask if "attention_mask" in inputs else None,
                    token_type_ids=None,
                    label=example.label
                ),
                "inputs_bt":
                DatasetInputFeature(
                    index = ex_index,
                    input_ids=inputs_bt.input_ids,
                    attention_mask=inputs_bt.attention_mask if "attention_mask" in inputs_bt else None,
                    token_type_ids=None,
                    label=example.label
                ),
                }
            )
        else:
            features.append(
                DatasetInputFeature(
                    index = ex_index,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask if "attention_mask" in inputs else None,
                    token_type_ids=None,
                    label=example.label
                )
            )
            

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: {}".format(f))

    return features
