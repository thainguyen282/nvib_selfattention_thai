#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# This handles the downloading, preprocessing and tokenization of the data
# It also is the interface between pytorch lightning and the data and organises the dataloaders

import os
import pickle
import re

import pytorch_lightning as pl
import requests
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_modules.DataCollatorForSeq2SeqWithClassification import (
    DataCollatorForSeq2SeqWithClassification,
)


def load_senteval(file):
    # Make directory if it does not exist
    if not os.path.exists("data/senteval"):
        os.mkdir("data/senteval")

    # Download the data if it does not exist
    if not os.path.exists(f"data/senteval/{file}.txt"):
        url = f"https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/{file}.txt"
        req = requests.get(url)
        with open(f"data/senteval/{file}.txt", "wb") as f:
            f.write(req.content)

    # Make a dictionary
    mode_dict = {"te": "test", "tr": "train", "va": "validation"}
    data_dict = {
        "train": {"text": [], "label": []},
        "test": {"text": [], "label": []},
        "validation": {"text": [], "label": []},
    }

    # Read the file
    with open(f"data/senteval/{file}.txt", "r") as f:
        for line in f:
            row = line.split("\t")
            # Get the mode
            data_dict[mode_dict[row[0]]]["label"].append(row[1])
            data_dict[mode_dict[row[0]]]["text"].append(row[2])

    return data_dict


def process_dataset(corpus, save_path, data_name, name):
    """
    Preprocess the data from a corpus to a cleaned dictionary with the text, lengths and total tokens
    :param corpus: the origonal corpus
    :param save_path: path for saving
    :return: dictionary with the text
    """
    if not os.path.exists(
        os.path.join(save_path, data_name, f"{name}preprocessed_text.pkl")
    ):

        def preprocess(sentence, label):
            # Clean the sentences
            # Remove everything but a-zA-Z0-9 or <> for <unk> or . ,
            sentence = re.sub("[^a-zA-Z0-9 \,'\<\>\.]", "", sentence)
            # Make lowercase
            sentence = sentence.lower()
            # Strip trailing white space
            sentence = sentence.strip()
            # Strip multiple white space
            sentence = re.sub(" +", " ", sentence)

            return sentence, label

        for subset in tqdm(corpus):
            print("Processing the subset: ", subset)
            print("Cleaning data:")
            # Preprocess the data in parallel
            lst_tuples = list(
                map(preprocess, tqdm(corpus[subset]["text"]), corpus[subset]["label"])
            )
            # Take list of tuples and make 2 lists of text and labels
            lst_lsts = list(map(list, tqdm(zip(*lst_tuples))))
            corpus[subset]["text"] = lst_lsts[0]
            corpus[subset]["label"] = lst_lsts[1]

        print("Saving:")
        file = open(
            os.path.join(save_path, data_name, f"{name}preprocessed_text.pkl"), "wb"
        )
        pickle.dump(corpus, file)
        file.close()
    else:
        print("Loading from disk")
        file = open(
            os.path.join(save_path, data_name, f"{name}preprocessed_text.pkl"), "rb"
        )
        corpus = pickle.load(file)
        file.close()

    data = DatasetDict(
        {
            "test": HFDataset.from_dict(corpus["test"]),
            "train": HFDataset.from_dict(corpus["train"]),
            "validation": HFDataset.from_dict(corpus["validation"]),
        }
    )

    return data


class DatasetWrapper(Dataset):
    """
    Dataset wrapper to set the __getitem__ function
    """

    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict

    def __len__(self):
        return len(self.dataset_dict["labels"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset_dict["input_ids"][idx],
            "labels": self.dataset_dict["labels"][idx],
            "attention_mask": self.dataset_dict["attention_mask"][idx],
        }


def prepare_tokenization(
    dataset,
    dataset_name,
    task_name,
    name,
    model_name,
    tokenizer,
    max_length=None,
    number_of_workers=1,
):
    """
    :param dataset: downloaded dataset from hugging face
    :param tokenizer: the tokenizer
    :param padding: Boolean to pad or not to pad
    :return: tokenized and padded model input
    """
    path = f"data/tokenized_{model_name}_{dataset_name}_{task_name}_{name}.pkl"
    # Train, validation or test
    dataset = dataset[name]
    column_names = dataset.column_names

    # if path does not exist
    if not os.path.exists(path):
        print("Tokenizing and saving to ", path)

        def tokenize_per_example(examples):
            # Tokenize into lists
            src = examples["text"]

            # Encoder inputs ... [SEP]
            model_inputs = tokenizer(src, truncation=True, max_length=max_length)
            model_inputs["input_ids"] = [
                input_ids[1:] for input_ids in model_inputs["input_ids"]
            ]
            model_inputs["attention_mask"] = [
                attn_mask[1:] for attn_mask in model_inputs["attention_mask"]
            ]

            # Labels ... [SEP] for reconstruction
            model_inputs["labels"] = model_inputs["input_ids"]
            # turn class labels into numbers
            class_dict = {c: i for i, c in enumerate(set(examples["label"]))}
            model_inputs["class"] = [class_dict[label] for label in examples["label"]]

            return model_inputs

        # Run function in parallel
        model_inputs = dataset.map(
            tokenize_per_example,
            batched=True,
            num_proc=number_of_workers,
            remove_columns=column_names,
            load_from_cache_file=False,  # This can be helpful but sometimes it gets stuck!
        )
        # Pickle save
        file = open(path, "wb")
        pickle.dump(model_inputs, file)
        file.close()


def load_prepared_data(name, data_name, task_name, model_name):
    path = f"data/tokenized_{model_name}_{data_name}_{task_name}_{name}.pkl"
    # Pickle pickle load
    file = open(path, "rb")
    model_inputs = pickle.load(file)
    file.close()

    return model_inputs


class SentEvalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        pl_model,
        batch_size,
        data,
        data_subset,
        num_workers,
        max_length,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = pl_model.tokenizer

        # Handles the padding and right shifting of decoder inputs
        self.collator = DataCollatorForSeq2SeqWithClassification(
            tokenizer=pl_model.tokenizer,
            model=pl_model.model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if "fp16" in kwargs else None,
        )

        self.batch_size = batch_size
        self.data = data
        self.data_subset = data_subset
        self.num_workers = num_workers
        self.max_length = max_length
        self.model_name = pl_model.model_type

    def prepare_data(self):
        # Download data or load data
        self.original_dataset = load_senteval(self.data_subset)

        # Preprocess data
        self.dataset = process_dataset(
            self.original_dataset, "data", self.data, self.data_subset
        )

        # Tokenize here and save to disk
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            task_name=self.data_subset,
            name="train",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            task_name=self.data_subset,
            name="validation",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            task_name=self.data_subset,
            name="test",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_inputs = load_prepared_data(
                name="train",
                data_name=self.data,
                task_name=self.data_subset,
                model_name=self.model_name,
            )

            self.validation_inputs = load_prepared_data(
                name="validation",
                # name="train",  # For overfitting
                data_name=self.data,
                task_name=self.data_subset,
                model_name=self.model_name,
            )
        # Assign validation dataset for use in dataloader(s)
        if stage == "validate":
            self.validation_inputs = load_prepared_data(
                name="validation",
                data_name=self.data,
                task_name=self.data_subset,
                model_name=self.model_name,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_inputs = load_prepared_data(
                name="test",
                data_name=self.data,
                task_name=self.data_subset,
                model_name=self.model_name,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_inputs,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.validation_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
