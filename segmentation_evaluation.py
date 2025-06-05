#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Set directory to root
import inspect
import os
import sys

import numpy as np
from nltk import word_tokenize
from scipy.optimize import linear_sum_assignment

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# Imports
import argparse
import os

from tqdm import tqdm

from data_modules.ReconstructionDataModule import ReconstructionDataModule
from models.nvib_sa_transformer import NVIBSaTransformerLightning
from models.transformer import TransformerLightning
from utils import *


def calc_f1(rr, rel, retr):
    """Calculate F1, precision, and recall"""
    if rr == 0:
        return 0, 0, 0
    p = rr / retr
    r = rr / rel
    return 2 * p * r / (p + r), p, r


def compute_segmentation_scores(ids, sentence):
    """
    Compute the segmentation scores between the words of the sentence and the discovered segments.
    It first uses Hungarian matching algorithm to find the most overlapping segments and words.
    Then it computes the retrieval measures, F1, Precision and Recall.
    Args:
        ids: the discovered segment ids extracted from the attention maps
        sentence: the sentence

    Returns: f1, precision, recall
    """
    f1s = []
    rs = []
    ps = []

    tokenize_sent = word_tokenize("".join(sentence))

    # Find the discovered spans from the ids
    b = 0  # beginning of the span
    e = 0  # end of the span
    discovered_spans = []
    for i, id in enumerate(ids[:-1]):
        if ids[i + 1] == id:
            e = i + 1
        else:
            discovered_spans.append((b, e))
            b = i + 1
            e = b
    # take care for the last token
    discovered_spans.append((b, e))

    # Find the real word spans
    word_spans = []
    c = 0
    for word in tokenize_sent:
        if c + len(word) < len(sentence) and sentence[c + len(word)] == " ":
            word_spans.append(
                (c, c + len(word) - 1)
            )  # remove -1 if you want to consider space
            c += len(word) + 1
        else:  # do not jump the space
            word_spans.append((c, c + len(word) - 1))
            c += len(word)

    # Find the best match between the discovered spans and the real word spans
    w = np.zeros((len(word_spans), len(discovered_spans)))
    for i, (sa, ea) in enumerate(word_spans):
        set1 = set(range(sa, ea + 1))
        for j, (sb, eb) in enumerate(discovered_spans):
            if ea < sb:
                break
            w[i, j] = len(
                set1.intersection(range(sb, eb + 1))
            )  # length of the longest common substring
    row_ind, col_ind = linear_sum_assignment(-w)  # Hungarian algorithm

    # Calculate the precisio, recall, and f1 score
    for r, c in zip(row_ind, col_ind):
        f1, p, r = calc_f1(
            w[r, c],
            word_spans[r][1] - word_spans[r][0] + 1,
            discovered_spans[c][1] - discovered_spans[c][0] + 1,
        )
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    return f1s, ps, rs


def get_segment_ids(
    batch,
    batch_item,
    tokenizer,
    attentions_all,
    attention_type,
    prior=None,
):
    attentions = attentions_all[attention_type]

    input_sentence = tokenizer.convert_ids_to_tokens(batch["input_ids"][batch_item, :])
    input_sentence = strip_after_token([input_sentence], tokenizer.sep_token)[0] + [
        "[S]"
    ]
    if prior is not None:
        input_sentence = [prior] + input_sentence

    if attention_type == "cross_attentions":
        output_ids = batch["decoder_input_ids"][batch_item, :]

    else:
        # Self attention or encoder attention
        output_ids = batch["input_ids"][batch_item, :]

    output_sentence = tokenizer.convert_ids_to_tokens(output_ids)
    output_sentence = strip_after_token([output_sentence], tokenizer.sep_token)[0]

    discovered_segment_ids = (
        torch.argmax(
            attentions[-1][batch_item, 0, :, :][
                : len(output_sentence), : len(input_sentence)
            ],
            dim=1,
        )
        .detach()
        .cpu()
    )

    return discovered_segment_ids, output_sentence


def main(args):
    dict_args = vars(args)
    OUTPUT_PATH = os.path.join(
        args.model_dir,
        args.trained_model_name
    )
    print("Output path: ", OUTPUT_PATH)
    # Best model
    # CHECKPOINT_PATH = get_best_model_path(OUTPUT_PATH)
    CHECKPOINT_PATH = get_checkpoint_path(OUTPUT_PATH)

    # Load model
    # Select pytorch lighting models
    model = {
        "Transformer": TransformerLightning,
        "NVIBSaTransformer": NVIBSaTransformerLightning,
    }[args.model]

    # Load model
    pl_model = load_model_strictly(CHECKPOINT_PATH, model, args)

    dm = ReconstructionDataModule(pl_model, **dict_args)
    dm.prepare_data()

    dm.setup("validate")
    pl_model.eval()
    model = pl_model.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)
    model.to(device)

    avg_prop_layer = [[], [], []]

    f1s = []
    rs = []
    ps = []

    for batch_idx, batch in enumerate(tqdm(dm.val_dataloader())):
        batch = batch.to(device)
        batch["input_ids"] = batch["input_ids"].transpose(0, 1).to(device)
        batch["decoder_input_ids"] = (
            batch["decoder_input_ids"].transpose(0, 1).to(device)
        )

        # Teacher forcing prediction
        model_outputs = model(**batch)

        # for i in range(3):
        #     avg_prop_layer[i].append(model_outputs['latent_dict_list'][i]['avg_prop_vec'])

        # Make batch first for the attention pattern
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        for i in range(0, len(batch["input_ids"])):
            segment_ids, output_sentence = get_segment_ids(
                batch,
                i,
                pl_model.tokenizer,
                model_outputs,
                "encoder_attentions",
                prior="[P]",
            )
            f1, p, r = compute_segmentation_scores(segment_ids, output_sentence)
            f1s.extend(f1)
            ps.extend(p)
            rs.extend(r)

    # Compute macro average of each of the metrics
    print(
        "Final macro F1 ",
        np.mean(f1s),
        "Final macro precision ",
        np.mean(ps),
        "Final macro recall",
        np.mean(rs),
    )

    # print(f"layer 4 {np.mean(avg_prop_layer[0])}")
    # print(f"layer 5 {np.mean(avg_prop_layer[1])}")
    # print(f"layer 6 {np.mean(avg_prop_layer[2])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths + Naming
    parser.add_argument(
        "--experiment_name",
        default="potential1lyr_model",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--project_name",
        default="recreate_improve",
        type=str,
        help="Project name for wandb",
    )
    parser.add_argument(
        "--output_dir", default="outputs", type=str, help="Output directory"
    )
    parser.add_argument("--model_dir", type=str, help="Learned model directory")
    parser.add_argument(
        "--trained_model_name", type=str, help="Trained Transformer model name"
    )
    # Data
    parser.add_argument("--data", type=str, default="wikitext", help="Dataset")
    parser.add_argument(
        "--data_subset",
        type=str,
        default="wikitext-2-raw-v1",
        help="subset of the dataset",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for processing"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum number of tokens of inputs and outputs",
    )

    # Model
    parser.add_argument(
        "--model",
        default="NVIBSaTransformer",
        help="Model selection",
    )

    # Training
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fp16", action="store_true", help="Use 16-bit precision")

    # Transformer
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=1, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=3, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=2, help="Number of decoder layers"
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=512, help="Feedforward dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")

    # Token deletion probability
    parser.add_argument(
        "--deletion_prob", type=float, default=0, help="Token deletion prob"
    )
    # Word span deletion probability
    parser.add_argument(
        "--deletion_type",
        type=str,
        default="token",
        choices=["token", "word", "token_word", "None"],
        help="Token deletion prob",
    )

    # NVIB
    parser.add_argument(
        "--num_nvib_encoder_layers",
        type=int,
        default=1,
        help="Number of nvib encoder layers",
    )
    parser.add_argument(
        "--kappa", type=float, default=1, help="number of samples for NVIB"
    )
    parser.add_argument("--delta", type=float, default=1, help="delta for NVIB")
    parser.add_argument(
        "--kld_lambda", type=float, default=0, help="KL dirichlet lambda"
    )
    parser.add_argument(
        "--klg_lambda", type=float, default=0, help="KL gaussian lambda"
    )
    parser.add_argument(
        "--kl_annealing_type",
        type=str,
        default="constant",
        help="KL annealing warm up type",
    )
    parser.add_argument(
        "--weighted_kl",
        action="store_true",
        help="weight the KL divergence - otherwise its averaged",
    )

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_steps", default=1, type=int)
    args = parser.parse_args()
    main(args)
