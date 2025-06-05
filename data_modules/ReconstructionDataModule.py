#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# This handles the downloading, preprocessing and tokenization of the data
# It also is the interface between pytorch lightning and the data and organises the dataloaders

import itertools
import os
import pickle
import re

import nltk
import numpy as np
import pytorch_lightning as pl
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.data import DataCollatorForSeq2Seq

from data_modules.DataCollatorForSeq2SeqWithNoise import DataCollatorForSeq2SeqWithNoise

nltk_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def split_sentences(text):
    sentences = nltk_tokenizer.tokenize(text)
    return sentences


def process_dataset(corpus, data_name, save_path):
    """
    Preprocess the data from a corpus to a cleaned dictionary with the text, lengths and total tokens
    :param corpus: the origonal corpus
    :param save_path: path for saving
    :return: dictionary with the text
    """
    if not os.path.exists(os.path.join(save_path, data_name, "preprocessed_text.pkl")):

        def preprocess(sentence):
            # Clean the sentences
            # Remove everything but a-zA-Z0-9 or <> for <unk> or . ,
            sentence = re.sub("[^a-zA-Z0-9 \,'\<\>\.]", "", sentence)
            # Make lowercase
            sentence = sentence.lower()
            # Strip trailing white space
            sentence = sentence.strip()
            # Strip multiple white space
            sentence = re.sub(" +", " ", sentence)
            return sentence

        text_dict = {}
        for subset in tqdm(corpus):
            print("Processing the subset: ", subset)
            print("Splitting data:")
            text = list(
                filter(
                    lambda x: (len(x) != 0 and len(x) != 1 and x[1] != "="),
                    tqdm(corpus[subset]["text"]),
                )
            )
            sentences = list(map(split_sentences, tqdm(text)))
            sentences = list(itertools.chain(*sentences))
            print("Cleaning data:")
            sentences = list(map(preprocess, tqdm(sentences)))
            text_dict[subset] = sentences
            print("Saving:")
            if not os.path.exists(os.path.join(save_path, data_name)):
                os.makedirs(os.path.join(save_path, data_name))
            with open(os.path.join(save_path, data_name, subset + ".csv"), "w") as f:
                for sentence in sentences:
                    f.write(sentence + "\n")

            file = open(
                os.path.join(save_path, data_name, "preprocessed_text.pkl"), "wb"
            )
            pickle.dump(text_dict, file)
            file.close()
    else:
        print("Loading from disk")
        file = open(os.path.join(save_path, data_name, "preprocessed_text.pkl"), "rb")
        text_dict = pickle.load(file)
        file.close()

    # Make it into a dataset dict
    data = DatasetDict(
        {
            "test": HFDataset.from_dict({"text": text_dict["test"]}),
            "train": HFDataset.from_dict({"text": text_dict["train"]}),
            "validation": HFDataset.from_dict({"text": text_dict["validation"]}),
        }
    )

    return data


def prepare_tokenization(
    dataset,
    dataset_name,
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
    path = f"data/tokenized_{tokenizer.name_or_path.split('/')[-1]}_{model_name}_{dataset_name}_{name}.pkl"
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


def load_prepared_data(tokenizer, name, data_name, model_name):
    path = f"data/tokenized_{tokenizer.name_or_path.split('/')[-1]}_{model_name}_{data_name}_{name}.pkl"
    # Pickle pickle load
    file = open(path, "rb")
    model_inputs = pickle.load(file)
    file.close()

    return model_inputs


class ReconstructionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        pl_model,
        batch_size,
        data,
        data_subset,
        num_workers,
        max_length,
        fp16,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = pl_model.tokenizer

        # Handles the padding and right shifting of decoder inputs AND ADDS NOISE
        self.noisy_collator = DataCollatorForSeq2SeqWithNoise(
            tokenizer=pl_model.tokenizer,
            model=pl_model.model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if fp16 else None,
            noise_prob=kwargs["deletion_prob"],
            noise_type=kwargs["deletion_type"],
        )

        self.collator = DataCollatorForSeq2Seq(
            tokenizer=pl_model.tokenizer,
            model=pl_model.model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if fp16 else None,
        )

        self.batch_size = batch_size
        self.data = data
        self.data_subset = data_subset
        self.num_workers = num_workers
        self.max_length = max_length
        self.model_name = pl_model.model_type

    def prepare_data(self):
        # Download data or load data
        self.original_dataset = load_dataset(self.data, self.data_subset)

        # Preprocess data
        self.dataset = process_dataset(self.original_dataset, self.data_subset, "data")

        # Tokenize here and save to disk
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="train",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="validation",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
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
                tokenizer=self.tokenizer,
                name="train",
                data_name=self.data,
                model_name=self.model_name,
            )

            self.validation_inputs = load_prepared_data(
                tokenizer=self.tokenizer,
                name="validation",
                # name="train",  # For overfitting
                data_name=self.data,
                model_name=self.model_name,
            )
        # Assign validation dataset for use in dataloader(s)
        if stage == "validate":
            self.validation_inputs = load_prepared_data(
                tokenizer=self.tokenizer,
                name="validation",
                data_name=self.data,
                model_name=self.model_name,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_inputs = load_prepared_data(
                tokenizer=self.tokenizer,
                name="test",
                data_name=self.data,
                model_name=self.model_name,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_inputs,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.noisy_collator,
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
