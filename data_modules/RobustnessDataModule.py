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
from textattack.augmentation import Augmenter, CharSwapAugmenter
from textattack.transformations import (
    WordSwapNeighboringCharacterSwap,  # Swap two neighboring characters
)
from textattack.transformations import (
    WordSwapRandomCharacterDeletion,  # Delete a random character
)
from textattack.transformations import (
    WordSwapRandomCharacterInsertion,  # Insert a random character
)
from textattack.transformations import (
    WordSwapRandomCharacterSubstitution,  # Substitute a random character
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.data import DataCollatorForSeq2Seq

# transformation = WordSwapNeighboringCharacterSwap()
# augmenter = Augmenter(transformation=transformation)
# s = 'I am fabulous.'
# augmenter.augment(s)


nltk_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def group_from_indecies(lst, indecies):
    "Group a list based on the indecies"
    size = len(lst)
    if len(indecies) != 0:
        groups = [
            lst[i:j]
            for i, j in zip(
                [0] + indecies,
                indecies + ([size] if indecies[-1] != size else []),
            )
        ]
    else:
        groups = [lst]
    return groups


def split_sentences(text):
    sentences = nltk_tokenizer.tokenize(text)
    return sentences


def process_dataset(corpus, data_name, save_path, transformation, transform_prob=0.0):
    """
    Preprocess the data from a corpus to a cleaned dictionary with the text, lengths and total tokens
    :param corpus: the origonal corpus
    :param save_path: path for saving
    :return: dictionary with the text
    """
    path = os.path.join(
        save_path, data_name, f"preprocessed_text_{transformation}{transform_prob}.pkl"
    )

    if transformation == "swap":
        transform = WordSwapNeighboringCharacterSwap()
    elif transformation == "delete":
        transform = WordSwapRandomCharacterDeletion()
    elif transformation == "insert":
        transform = WordSwapRandomCharacterInsertion()
    elif transformation == "substitute":
        transform = WordSwapRandomCharacterSubstitution()
    else:
        transform = None
    # If a specific transformation is desired then use this. Otherwise, use the default
    # augmenter = Augmenter(transformation=transform, transformations_per_example=1, pct_words_to_swap=transform_prob,
    #                       high_yield=True,
    #                       fast_augment=True)

    augmenter = CharSwapAugmenter(
        transformations_per_example=2,
        pct_words_to_swap=transform_prob,
        high_yield=True,
        fast_augment=True,
    )

    def perturb(sentence):
        perturbed_sentence = augmenter.augment(sentence)[0]
        # print(sentence, perturbed_sentence)
        return (
            perturbed_sentence.lower()
        )  # Not sure to include this or not because the library inserts majuscules as well

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
        # Apply transform
        # loops = np.sum(np.random.rand(len(sentence)) < transform_prob)
        # for _ in range(loops):
        #     # Apply the transformation
        #     sentence = augmenter.augment(sentence)[0]
        return sentence

    if not os.path.exists(path):
        text_dict = {}
        for subset in tqdm(corpus):
            text_dict[subset] = {}
            if subset != "train":
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
                print("Cleaning + Augmenting data:")
                sentences = list(
                    map(preprocess, tqdm(sentences))
                )  # we can reduce the data size here
                perturbed_sentences = (
                    list(map(perturb, tqdm(sentences)))
                    if transform_prob > 0
                    else list(map(lambda x: x, tqdm(sentences)))
                )

                text_dict[subset]["original"] = sentences
                text_dict[subset]["perturbed"] = perturbed_sentences

                # Make it into a dataset dict
        data = DatasetDict(
            {
                "test": HFDataset.from_dict(
                    {
                        "text": text_dict["test"]["perturbed"],
                        "label": text_dict["test"]["original"],
                    }
                ),
                "validation": HFDataset.from_dict(
                    {
                        "text": text_dict["validation"]["perturbed"],
                        "label": text_dict["validation"]["original"],
                    }
                ),
            }
        )
        print("Saving:")
        if not os.path.exists(os.path.join(save_path, data_name)):
            os.makedirs(os.path.join(save_path, data_name))

        file = open(path, "wb")
        pickle.dump(data, file)
        file.close()
    else:
        print("Loading from disk")
        file = open(path, "rb")
        data = pickle.load(file)
        file.close()

    return data


def prepare_tokenization(
    dataset,
    name,
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

    # Train, validation or test
    dataset = dataset[name]
    column_names = dataset.column_names

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

        tgt = examples["label"]
        decoder_inputs = tokenizer(tgt, truncation=True, max_length=max_length)
        # Labels ... [SEP] for reconstruction
        model_inputs["labels"] = [
            input_ids[1:] for input_ids in decoder_inputs["input_ids"]
        ]
        return model_inputs

    # Run function in parallel

    model_inputs = dataset.map(
        tokenize_per_example,
        batched=True,
        num_proc=number_of_workers,
        remove_columns=column_names,
        load_from_cache_file=False,  # This can be helpful but sometimes it gets stuck!
    )

    return model_inputs


class RobustnessDataModule(pl.LightningDataModule):
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

        # Handles the padding and right shifting of decoder inputs
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
        self.transformation_prob = kwargs["transformation_prob"]
        self.transformation = kwargs["transformation"]

    def prepare_data(self):
        # Download data or load data
        self.original_dataset = load_dataset(self.data, self.data_subset)

        # Preprocess data
        self.dataset = process_dataset(
            self.original_dataset,
            self.data_subset,
            "data",
            self.transformation,
            self.transformation_prob,
        )

        # Tokenize data
        self.train_inputs = prepare_tokenization(
            dataset=self.dataset,
            name="validation",
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )
        self.validation_inputs = prepare_tokenization(
            dataset=self.dataset,
            name="test",
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            pass
        # Assign validation dataset for use in dataloader(s)
        if stage == "validate":
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

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
