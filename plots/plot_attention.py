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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# Imports
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from tqdm import tqdm

from data_modules.PlotReconstructionDataModule import PlotReconstructionDataModule
from models.nvib_sa_transformer import NVIBSaTransformerLightning
from models.transformer import TransformerLightning
from utils import *


def attention_plot(
    batch,
    batch_item,
    tokenizer,
    attentions_all,
    attention_type,
    zmax=None,
    prior=None,
    num_heads=1,
    num_layers=1,
    save_path=None,
):
    attentions = attentions_all[attention_type]

    if attention_type == "cross_attentions":
        input_sentence = tokenizer.convert_ids_to_tokens(
            batch["input_ids"][batch_item, :]
        )
        input_sentence = strip_after_token([input_sentence], tokenizer.sep_token)[0] + [
            "[S]"
        ]
        if prior is not None:
            input_sentence = [prior] + input_sentence
        output_ids = batch["decoder_input_ids"][batch_item, :]

        output_sentence = tokenizer.convert_ids_to_tokens(output_ids)
        output_sentence = strip_after_token([output_sentence], tokenizer.sep_token)[0]

    else:
        # Self attention or encoder attention
        input_sentence = tokenizer.convert_ids_to_tokens(
            batch["input_ids"][batch_item, :]
        )
        input_sentence = strip_after_token([input_sentence], tokenizer.sep_token)[0] + [
            "[S]"
        ]
        if prior is not None:
            input_sentence = [prior] + input_sentence
        output_ids = batch["input_ids"][batch_item, :]

        output_sentence = tokenizer.convert_ids_to_tokens(output_ids)
        output_sentence = strip_after_token([output_sentence], tokenizer.sep_token)[
            0
        ] + ["[S]"]

    # Plot layers and heads
    # Num of encoder layers is num_layers
    # nvib_num_layers = len(attentions) - num_layers
    num_layers = len(attentions)
    for layer in range(num_layers):
        cross_attentions = attentions[layer][batch_item, 0, :, :]
        # cross_attentions = attentions[num_layers + layer][batch_item, 0, :, :]

        # Plot with matplotlib
        ax, fig = plt.subplots()
        g = sns.heatmap(
            cross_attentions[: len(output_sentence), : len(input_sentence)]
            .detach()
            .cpu(),
            yticklabels=output_sentence,
            xticklabels=input_sentence,
            cmap="viridis",
            linewidths=0.1,
            vmax=zmax,
            vmin=0,
            linecolor="black",
            cbar=False,
        )
        # Long len is around 60  font 5
        # Short len is around 10  font 10
        font = -(1 / 10) * len(output_sentence) + 11

        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(
                f"{save_path}/{attention_type}_{batch_item}_layer_{layer}.pdf",
                bbox_inches="tight",
            )
            plt.close()


def main(args):
    dict_args = vars(args)
    OUTPUT_PATH = os.path.join(
        args.output_dir,
        args.project_name,
        args.experiment_name,
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

    # Instantiate or load model
    pl_model, wandb_id = create_or_load_model(OUTPUT_PATH, CHECKPOINT_PATH, model, args)

    # Put in custom sentences
    custom_dataset = DatasetDict(
        {
            "test": HFDataset.from_dict({"text": [""]}),
            "train": HFDataset.from_dict({"text": [""]}),
            "validation": HFDataset.from_dict(
                {
                    "text": [
                        "In life , the lobsters are blue.",
                        "I think therefore I am.",  # REALLY NICE
                        "Wow, it's abstracting.",
                        "I am a cat.",
                        "That’s one small step for a man, a giant leap for mankind.",  # Less good
                        "I took the one less travelled by, and that has made all the difference.",
                        "Whatever you are, be a good one.",
                        "You must be the change you wish to see in the world.",
                    ]
                }
            ),
        }
    )

    # Make data module
    dm = PlotReconstructionDataModule(pl_model, dataset=custom_dataset, **dict_args)

    dm.prepare_data()

    dm.setup("validate")
    pl_model.eval()
    model = pl_model.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)
    model.to(device)

    for batch_idx, batch in enumerate(tqdm(dm.val_dataloader())):
        batch = batch.to(device)
        batch["input_ids"] = batch["input_ids"].transpose(0, 1).to(device)
        batch["decoder_input_ids"] = (
            batch["decoder_input_ids"].transpose(0, 1).to(device)
        )

        # Teacher forcing prediction
        model_outputs = model(**batch)

        # Make batch first for plotting
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        # Only the first batch and the first item of the batch
        for i in range(0, 8):
            batch_item = i
            # Plot attention
            attention_plot(
                batch,
                batch_item,
                pl_model.tokenizer,
                model_outputs,
                "encoder_attentions",
                zmax=1,
                prior="[P]",
                num_heads=model.nhead,
                num_layers=model.num_encoder_layers,
                save_path=OUTPUT_PATH,
            )

            attention_plot(
                batch,
                batch_item,
                pl_model.tokenizer,
                model_outputs,
                "cross_attentions",
                zmax=1,
                prior=None,  # Only Self attention has the prior!
                num_heads=model.nhead,
                num_layers=model.num_decoder_layers,
                save_path=OUTPUT_PATH,
            )


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

    # Data
    parser.add_argument(
        "--data", type=str, default="custom", help="Dataset, other options are wmt16"
    )
    parser.add_argument(
        "--data_subset", type=str, default="v1", help="subset of the dataset"
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

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_steps", default=1, type=int)
    args = parser.parse_args()
    main(args)

# python plots/plot_attention.py


# In life , the lobsters are blue.
# Homarus gammarus , known as the European lobster .
# Like other crustaceans , lobsters have a hard exoskeleton .
# Lobsters are mostly fished using lobster pots .
# He subsequently passed the course with a special distinction .

# python plots/plot_attention.py --experiment_name potential3lyr_delta0.25 --project_name recreate_improve --model NVIBSaTransformer --batch_size 512 --num_workers 1 --data wikitext --data_subset wikitext-2-raw-v1 --seed 42 --d_model 512 --nhead 1 --dim_feedforward 512 --num_encoder_layers 3 --num_decoder_layers 2 --dropout 0.1 --max_steps 8000 --deletion_prob 0.1 --deletion_type token --num_nvib_encoder_layers 3 --klg_lambda 0.01 --kld_lambda 1 --delta 0.25 --fp16 --weighted_kl --kl_annealing_type linear
