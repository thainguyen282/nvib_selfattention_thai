#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import itertools
from typing import Any, Optional, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.data import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PaddingStrategy


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


def word_span_deletion(examples, word_deletion_prob):
    # Tokenize into lists
    srcs = examples["input_ids"]
    masks = examples["attention_mask"]
    token_type_ids = examples["token_type_ids"]

    # Find the start of words
    start_indecies = list(i + 1 for i in range(len(srcs)) if srcs[i] == 101)

    # Group words together
    src_groups = group_from_indecies(srcs, start_indecies)
    mask_groups = group_from_indecies(masks, start_indecies)
    token_type_ids_groups = group_from_indecies(token_type_ids, start_indecies)

    # Create word mask but dont mask the last word eos token
    bool_mask = list(np.random.sample(len(start_indecies)) > word_deletion_prob) + [
        True
    ]

    # Filter out the words
    remaining_src_group = list(
        src_groups[i] for i in range(len(src_groups)) if bool_mask[i]
    )
    remaining_mask_group = list(
        mask_groups[i] for i in range(len(mask_groups)) if bool_mask[i]
    )
    remaining_token_type_ids_group = list(
        token_type_ids_groups[i]
        for i in range(len(token_type_ids_groups))
        if bool_mask[i]
    )

    # Flatten the lists
    examples["input_ids"] = list(itertools.chain(*remaining_src_group))
    examples["attention_mask"] = list(itertools.chain(*remaining_mask_group))
    examples["token_type_ids"] = list(itertools.chain(*remaining_token_type_ids_group))

    return examples


def token_deletion(examples, token_deletion_prob):
    # Tokenize into lists
    src = examples["input_ids"]
    mask = examples["attention_mask"]
    token_type_ids = examples["token_type_ids"]

    # Make list of boolean masks (Falses we keep True we delete)
    bool_mask = np.random.sample((len(mask))) < token_deletion_prob
    # Not spaces
    space_mask = np.array(src) != 101
    # Not last token
    last_token_mask = np.array(src) != 1

    # Reverse so Trues we keep False we delete
    bool_mask = list(~(bool_mask & space_mask & last_token_mask))

    # Delete tokens: Zip together and filter
    src = list(b for a, b in zip(bool_mask, src) if a)

    mask = list(b for a, b in zip(bool_mask, mask) if a)
    token_type_ids = list(b for a, b in zip(bool_mask, token_type_ids) if a)

    examples["input_ids"] = src
    examples["attention_mask"] = mask
    examples["token_type_ids"] = token_type_ids
    return examples


class DataCollatorForSeq2SeqWithNoise(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __init__(self, noise_prob, noise_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_prob = noise_prob
        self.noise_type = noise_type

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)
        # Features is a list of dictionaries
        # Each dictionary has keys: input_ids, attention_mask, token_type_ids, labels
        # make noise_prob a list
        if not isinstance(self.noise_prob, list):
            self.noise_prob = [self.noise_prob] * len(features)

        # Apply noise
        if self.noise_type == "token":
            features = list(map(token_deletion, features, self.noise_prob))
        elif self.noise_type == "word":
            features = list(map(word_span_deletion, features, self.noise_prob))
        elif self.noise_type == "token_word":
            features = list(map(word_span_deletion, features, self.noise_prob))
            features = list(map(token_deletion, features, self.noise_prob))
        else:
            pass

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
