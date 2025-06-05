#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_modules.ReconstructionDataModule import ReconstructionDataModule
from models.nvib_sa_transformer import NVIBSaTransformerLightning
from models.transformer import TransformerLightning
from utils import *

# Stops an annoying warning from transformers
# You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def main(args):
    load_dotenv()
    START_TIME = datetime.now().replace(microsecond=0)
    dict_args = vars(args)
    pl.seed_everything(args.seed)
    OUTPUT_PATH = os.path.join(args.output_dir, args.project_name, args.experiment_name)
    CHECKPOINT_PATH = get_checkpoint_path(OUTPUT_PATH)
    BEST_MODEL_PATH = get_best_model_path(OUTPUT_PATH)

    # Select pytorch lighting models
    model = {
        "Transformer": TransformerLightning,
        "NVIBSaTransformer": NVIBSaTransformerLightning,
    }[args.model]

    # Instantiate or load model
    model, wandb_id = create_or_load_model(OUTPUT_PATH, CHECKPOINT_PATH, model, args)
    model = load_model(BEST_MODEL_PATH, model, args)

    # Make data module
    dm = ReconstructionDataModule(model, **dict_args)

    # Checkpointing callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_PATH, every_n_train_steps=args.checkpoint_interval
    )
    bestmodel_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        every_n_train_steps=args.checkpoint_interval,
        dirpath=OUTPUT_PATH,
        filename="best_model",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # WandB logger
    wandb_logger = WandbLogger(
        project=args.project_name, entity=args.entity, id=wandb_id, log_model="None"
    )

    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        # limit_train_batches=0.1,
        # limit_val_batches=0.01,
        # limit_test_batches=0.1,
        # overfit_batches=1,
        # deterministic=True,
        accelerator="auto",
        gradient_clip_val=0.1,
        precision=16 if args.fp16 else 32,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.validation_interval,
        callbacks=[checkpoint_callback, bestmodel_callback, lr_monitor],
        logger=wandb_logger,
        num_sanity_val_steps=0,
        fast_dev_run=args.fast_dev_run,
    )

    # Dev set evaluation
    trainer.validate(model=model, datamodule=dm)

    # Test set evaluation
    trainer.test(model=model, datamodule=dm)


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
        default="local_experiments",
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

    # Training
    parser.add_argument("--seed", default=42, type=int)
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
    # parser.add_argument("--batch_first", action="store_true", help="Normalize before")
    # parser.add_argument("--norm_first", action="store_true", help="Normalize after")

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
    parser.add_argument("--batch_size", default=1, type=int)

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
