#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import argparse
import os

import evaluate
import numpy as np
import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch.nn as nn
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

import wandb
from data_modules.RobustnessDataModule import RobustnessDataModule
from models.nvib_sa_transformer import NVIBSaTransformerLightning
from models.transformer import TransformerLightning
from utils import *

load_dotenv()

# Stops an annoying warning from transformers
# You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def main(args):
    pl.seed_everything(args.seed)
    OUTPUT_PATH = os.path.join(
        args.output_dir,
        args.project_name,
        args.experiment_name,
    )
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    print(args)
    if args.checkpoint == "last":
        MODEL_PATH = get_checkpoint_path(
            os.path.join(args.model_dir, args.trained_model_name)
        )
    else:
        MODEL_PATH = get_best_model_path(
            os.path.join(args.model_dir, args.trained_model_name)
        )

    # Select pytorch lighting models
    pl_model = {
        "Transformer": TransformerLightning,
        "NVIBSaTransformer": NVIBSaTransformerLightning,
    }[args.model]

    # Instantiate or load model
    pl_model = load_model_strictly(MODEL_PATH, pl_model, args)

    # WandB logger - Includes the entity (which is the team name)
    wandb_logger = WandbLogger(
        project=args.project_name, entity=args.entity, log_model="None"
    )
    wandb_logger.experiment.config.update(args)
    wandb.define_metric("custom_step")
    wandb.define_metric("self chrf", step_metric="custom_step")
    wandb.define_metric("self bleu", step_metric="custom_step")
    wandb.define_metric("ce", step_metric="custom_step")

    deletion_probs = np.arange(0.0, 1.01, 0.2)
    bleu_scores = []
    chrf_scores = []
    ce_scores = []
    step = 0
    for deletion_prob in tqdm(deletion_probs):
        print(f"Transformation probability: {deletion_prob}")
        dm = RobustnessDataModule(
            pl_model=pl_model,
            data=args.data,
            data_subset=args.data_subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            fp16=args.fp16,
            transformation_prob=deletion_prob,
            transformation=args.transformation,
        )

        dm.prepare_data()

        pl_model.eval()
        model = pl_model.model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl_model.to(device)
        model.to(device)

        predictions = []
        tgts = []
        ce = []
        for batch_idx, batch in enumerate(tqdm(dm.val_dataloader())):
            batch = batch.to(device)
            batch["input_ids"] = batch["input_ids"].transpose(0, 1).to(device)
            batch["decoder_input_ids"] = (
                batch["decoder_input_ids"].transpose(0, 1).to(device)
            )

            # Teacher forcing prediction
            model_outputs = model(**batch)
            criterion = nn.CrossEntropyLoss(
                ignore_index=pl_model.tokenizer.pad_token_id, reduction="mean"
            )
            targets = torch.flatten(batch["labels"].transpose(0, 1))  # [Nt x B]
            logits = torch.flatten(
                model_outputs["logits"], start_dim=0, end_dim=1
            )  # [Nt x B, V]
            cross_entropy_loss = criterion(logits, targets)  # [Nt x B]
            ce.append(cross_entropy_loss.item())

            # Autoregressive prediction
            generated_ids = model.generate(
                batch["input_ids"],
                max_new_tokens=256,
            )
            batch_predictions = pl_model.tokenizer.batch_decode(
                generated_ids.transpose(0, 1)
            )
            batch_predictions = strip_after_token(
                batch_predictions, pl_model.tokenizer.sep_token
            )
            predictions.extend(batch_predictions)

            tgt = pl_model.tokenizer.batch_decode(batch["labels"])
            tgt = strip_after_token(tgt, pl_model.tokenizer.sep_token)
            tgts.extend(tgt)

        chrf_score = evaluate.load("chrf")
        bleu_score = evaluate.load("sacrebleu")

        bleu_results = bleu_score.compute(
            predictions=predictions, references=[[ref] for ref in tgts]
        )
        bleu_scores.append(bleu_results["score"])

        chrf_results = chrf_score.compute(
            predictions=predictions, references=[[ref] for ref in tgts]
        )
        chrf_scores.append(chrf_results["score"])
        ce_scores.append(np.mean(ce))

        print(f"CE given char deletion of {deletion_prob}: {np.mean(ce)}")
        wandb.log({f"CE_{deletion_prob}": np.mean(ce)})

        print(f"BLEU score given char deletion of {deletion_prob}: {bleu_scores[-1]}")
        wandb.log({f"selfBLEU_{deletion_prob}": bleu_scores[-1]})
        print(
            f"CHRF score given character deletion of {deletion_prob}: {chrf_scores[-1]}"
        )
        wandb.log({f"selfCHRF_{deletion_prob}": chrf_scores[-1]})
        wandb.log(
            {
                "self chrf": chrf_scores[-1],
                "custom_step": step,
                "self bleu": bleu_scores[-1],
                "ce": np.mean(ce),
            }
        )
        step += 1
        # wandb.log({"self bleu": bleu_scores[-1]})
        # wandb.log({"ce": np.mean(ce)})
        # breakpoint()

    # Create pandas dataframe
    df = pd.DataFrame(
        {
            "deletion_prob": deletion_probs,
            "bleu_score": bleu_scores,
            "chrf_score": chrf_scores,
            "reconstruction_cross_entropy": ce_scores,
        }
    )

    # save dataframe
    df.to_csv(
        os.path.join(OUTPUT_PATH, f"scores_deletion_probs_{args.experiment_name}.csv")
    )
    # Plot the results
    fig = px.line(
        df,
        x="deletion_prob",
        y="bleu_score",
        title="BLEU score vs. deletion probability",
    )
    wandb.log({"plot bleu score vs deletion prob": fig})
    fig = px.line(
        df,
        x="deletion_prob",
        y="chrf_score",
        title="CHRF score vs. deletion probability",
    )
    wandb.log({"plot chrf score vs deletion prob": fig})
    fig = px.line(
        df,
        x="deletion_prob",
        y="reconstruction_cross_entropy",
        title="CE score vs. deletion probability",
    )
    wandb.log({"plot cross entropy score vs deletion prob": fig})
    # fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths + Naming
    parser.add_argument(
        "--experiment_name",
        default="initial_experiment",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--project_name",
        default="robustness_evaluation",
        type=str,
        help="Project name for wandb",
    )
    parser.add_argument(
        "--output_dir", default="outputs", type=str, help="Output directory"
    )
    parser.add_argument("--entity", type=str, help="Wandb entity")

    # Data
    parser.add_argument(
        "--data", type=str, default="wikitext", help="Dataset, other options are wmt16"
    )
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
        default="Transformer",
        help="Model selection",
    )
    parser.add_argument("--model_dir", type=str, help="Learned model directory")
    parser.add_argument(
        "--trained_model_name", type=str, help="Trained Transformer model name"
    )
    # Training
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--checkpoint",
        default="best",
        type=str,
        choices=["best", "last"],
        help="model checkpoint to load",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Touches all train, validation and test scripts for debugging",
    )
    parser.add_argument("--fp16", action="store_true", help="Use 16-bit precision")

    # Transformer
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=3, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=1, help="Number of decoder layers"
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=256, help="Feedforward dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    # Transformation + probability
    parser.add_argument(
        "--transformation_prob", type=float, default=0, help="Token deletion prob"
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default="delete",
        choices=["delete", "insert", "swap", "substitute"],
        help="type of transformation to apply to mimic the noise",
    )

    # NVIB
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
        "--weighted_kl",
        action="store_true",
        help="weight the KL divergence - otherwise its averaged",
    )

    # Learning rate + Schedulers
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--lr_scheduler", action="store_true", help="Use lr scheduler")
    parser.add_argument(
        "--perc_warmup",
        type=float,
        default=0,
        help="Learning rate warm up steps percentage",
    )
    parser.add_argument("--batch_size", default=512, type=int)

    # PL Trainer
    parser.add_argument(
        "--max_time",
        default=None,
        help="Max amount of time example - 00:02:45:00 for short gpus",
    )
    parser.add_argument(
        "--max_steps", default=1000, type=int, help="Max number of steps for training"
    )
    parser.add_argument(
        "--max_epochs", default=None, type=int, help="Max number of epochs for training"
    )
    # NOTE: gradient accumulation affects the checkpoint, if its 2 then it will checkpoint 2x less often
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="Gradient accumulation for models",
    )
    parser.add_argument(
        "--checkpoint_interval",
        default=100,
        type=int,
        help="Checkpointing every N training steps",
    )
    parser.add_argument(
        "--validation_interval",
        default=None,
        type=int,
        help="Do validation every N training steps",
    )

    args = parser.parse_args()

    main(args)
