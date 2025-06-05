#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import argparse
from datetime import datetime

import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_modules.ArxivDataModule import ArxivDataModule
from evaluation.probe import (
    AggregatingProbe,
    AttentionBasedClassifier,
    LitProbe,
)
from models.nvib_sa_transformer import NVIBSaTransformerLightning
from models.transformer import TransformerLightning
from utils import *

load_dotenv()

# Stops an annoying warning from transformers
# You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

num_classes = 20


def main(args):
    dict_args = vars(args)
    pl.seed_everything(args.seed)
    OUTPUT_PATH = os.path.join(
        args.output_dir,
        args.project_name,
        args.experiment_name,
        args.subject,
        args.dataset_size,
        args.trained_model_name,
        "layer_" + str(args.probe_layer_num),
        args.probe_model,
        args.probe_aggregation_fn,
        "masked",
        "is_weighted" + str(args.probe_use_pi),
        str(args.probe_hidden_dim),
        str(args.learning_rate),
        args.checkpoint,
    )
    if args.checkpoint == "last":
        MODEL_PATH = get_checkpoint_path(
            os.path.join(args.model_dir, args.trained_model_name)
        )
    else:
        MODEL_PATH = get_best_model_path(
            os.path.join(args.model_dir, args.trained_model_name)
        )
    CHECKPOINT_PATH = get_checkpoint_path(OUTPUT_PATH)
    START_TIME = datetime.now().replace(microsecond=0)

    # Select pytorch lighting models
    transformer_model = {
        "Transformer": TransformerLightning,
        "NVIBSaTransformer": NVIBSaTransformerLightning,
    }[args.model]

    # Instantiate or load model
    transformer_model = load_model_strictly(MODEL_PATH, transformer_model, args)
    # Make data module
    dm = ArxivDataModule(transformer_model, **dict_args)
    # Make the probing classifier
    probe_model = {
        "attn-based": AttentionBasedClassifier,
        "aggr-based": AggregatingProbe,
    }[args.probe_model]

    aggr = {"mean": torch.mean, "sum": torch.sum, "max": torch.max, "attn": None}[
        args.probe_aggregation_fn
    ]

    probe_classifier = probe_model(
        dim=transformer_model.model.d_model,
        n_output=num_classes,
        aggregation=aggr,
        use_pi=args.probe_use_pi,
        hidden_dim=args.probe_hidden_dim,
    )

    # Checkpointing callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_PATH, every_n_train_steps=args.checkpoint_interval
    )
    bestmodel_callback = ModelCheckpoint(
        monitor="val_acc_epoch",
        mode="max",
        every_n_train_steps=args.checkpoint_interval,
        dirpath=OUTPUT_PATH,
        filename="best_model",
    )

    # WandB logger - Includes the entity (which is the team name)
    wandb_logger = WandbLogger(
        project=args.project_name, entity=args.entity, log_model="None"
    )
    wandb_logger.experiment.config.update(args)

    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        gradient_clip_val=0.1,
        val_check_interval=args.validation_interval,
        callbacks=[bestmodel_callback, checkpoint_callback],
        logger=wandb_logger,
        num_sanity_val_steps=0,
    )

    model = LitProbe(transformer_model, probe_classifier, args, num_classes=num_classes)

    # Train model
    print("Start of training:", (datetime.now().replace(microsecond=0) - START_TIME))
    trainer.fit(model, datamodule=dm, ckpt_path=CHECKPOINT_PATH)
    print("End of training:", (datetime.now().replace(microsecond=0) - START_TIME))

    # Dev set evaluation
    trainer.validate(model, datamodule=dm, ckpt_path=get_best_model_path(OUTPUT_PATH))

    # Test set evaluation
    trainer.test(model, datamodule=dm, ckpt_path=get_best_model_path(OUTPUT_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths + Naming
    parser.add_argument(
        "--experiment_name", default="test_experiment", type=str, help="Experiment name"
    )
    parser.add_argument(
        "--project_name",
        default="arxiv_experiments",
        type=str,
        help="Project name for wandb",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Learned model directory",
    )
    parser.add_argument(
        "--trained_model_name",
        type=str,
        help="Trained Transformer model name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="Wandb entity (team) name",
    )
    # Data
    parser.add_argument("--data", type=str, default="arxiv", help="Dataset")
    parser.add_argument(
        "--subject",
        type=str,
        default="maths",
        choices=["maths", "cs", "physics"],
        help="subset of the dataset",
    )
    parser.add_argument(
        "--dataset_size", type=str, default="small", help="Dataset size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for processing"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum number of tokens of inputs and outputs",
    )

    # Model
    parser.add_argument(
        "--model",
        default="Transformer",
        help="Model selection",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--checkpoint",
        default="best",
        type=str,
        choices=["best", "last"],
        help="model checkpoint to load",
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

    # Transformer
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=1, help="Number of attention heads"
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

    parser.add_argument("--batch_size", default=256, type=int)
    # parser.add_argument(
    #     "--max_steps", default=1000, type=int, help="Max number of steps for training"
    # )
    parser.add_argument(
        "--max_epochs", default=50, type=int, help="Max number of epochs for training"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--validation_interval",
        default=0.25,
        type=float,
        help="Do validation every N training steps",
    )
    parser.add_argument(
        "--checkpoint_interval",
        default=100,
        type=int,
        help="Checkpointing every N training steps",
    )
    # probe parameters
    parser.add_argument(
        "--probe_model", default="attn-based", type=str, help="Probe model to use"
    )
    parser.add_argument(
        "--probe_aggregation_fn",
        default="attn",
        type=str,
        help="Aggregation function to apply to the inputs in the Aggregation  probe",
    )
    parser.add_argument(
        "--probe_layer_num",
        default=-1,
        type=int,
        help="Transformer layer number to probe",
    )
    parser.add_argument(
        "--probe_use_pi", default=False, type=bool, help="Use pi in probing"
    )
    parser.add_argument(
        "--probe_hidden_dim",
        default=256,
        type=int,
        help="Probing classifier hidden dimension",
    )
    args = parser.parse_args()
    main(args)
