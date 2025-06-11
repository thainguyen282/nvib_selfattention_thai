#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

from itertools import chain

import evaluate
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from utils import show_attention, strip_after_token


class kl_annealing:
    def __init__(
        self,
        end_of_warmup,
        wait_before_warmup=0,
        annealing_value_start=0,
        annealing_value_end=1,
        type="linear",
    ):
        self.annealing_value_start = annealing_value_start
        self.annealing_value_end = annealing_value_end
        self.end_of_warmup = end_of_warmup
        self.type = type
        self.wait_before_warmup = wait_before_warmup

    def __call__(self, step):
        # Linear annealing
        if self.type == "linear":
            if step < self.wait_before_warmup:
                return self.annealing_value_start
            elif step < self.end_of_warmup:
                return (step - self.wait_before_warmup) / (
                    self.end_of_warmup - self.wait_before_warmup
                )
            else:
                return self.annealing_value_end

        # # TODO: sin annealing
        # elif self.type == "cosine":
        #     if step < self.wait_before_warmup:
        #         return self.annealing_value_start
        #     elif step < self.end_of_warmup:
        #         return (
        #             math.cos(
        #                 (step - self.wait_before_warmup)
        #                 / (self.end_of_warmup - self.wait_before_warmup)
        #                 * math.pi
        #             )
        #             + 1
        #         ) / 2

        #     else:
        #         return self.annealing_value_end

        else:
            # Constant
            return self.annealing_value_end


def weighted_mean(kl_list, weighted_mean=False):
    if weighted_mean:
        weights = [i for i in range(1, len(kl_list) + 1)]
        # weights = [(2**i) for i in range(0, len(kl_list))]
    else:  # Equal weighted Mean
        weights = [1 for i in range(0, len(kl_list))]

    weights = [weight / (sum(weights)) for weight in weights]
    return sum([torch.mean(kl_layer) * weights[i] for i, kl_layer in enumerate(kl_list)])


class Seq2SeqLightning(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()

        # Training parameters
        self.learning_rate = args.learning_rate if hasattr(args, "learning_rate") else None
        self.learning_rate_scheduler = (
            args.lr_scheduler if hasattr(args, "lr_scheduler") else False
        )
        self.perc_warmup = args.perc_warmup if hasattr(args, "perc_warmup") else None

        # For NVIB models
        self.is_nvib = False
        self.weighted_kl = args.weighted_kl if hasattr(args, "weighted_kl") else False

        # Logging metrics
        self.log_bleu = False
        self.log_rouge = False
        self.log_chrf = False

        # Plotting attention
        self.plot_cross_attention = False
        self.plot_encoder_attention = False

        # For NVIB models
        self.kl_annealing_scheduler = (
            kl_annealing(
                annealing_value_start=0,
                annealing_value_end=1,
                wait_before_warmup=args.max_steps * 0.3,  # wait for 30% of the steps
                end_of_warmup=args.max_steps * 0.6,  # warmup
                type=args.kl_annealing_type,
            )
            if hasattr(args, "kl_annealing_type")
            else None
        )

    def training_step(self, batch, batch_idx):
        # print(f"training step batch_idx {batch_idx}, items: {batch['input_ids']}")
        # Correct orientation Nt x B
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["labels"] = batch["labels"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        # Forward pass
        model_outputs = self.model(**batch)
        # Get loss and ignore the pads
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="mean")
        # Transform targets
        targets = torch.flatten(batch["labels"])  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(model_outputs["logits"], start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]

        if self.is_nvib:
            # KL annealing
            kl_factor = self.kl_annealing_scheduler(self.global_step)
            self.log("kl_annealing_factor", float(kl_factor))

            kld = (
                weighted_mean(model_outputs["kl_dirichlet"], self.weighted_kl)
                * self.lambda_kld
                * kl_factor
            )
            klg = (
                weighted_mean(model_outputs["kl_gaussian"], self.weighted_kl)
                * self.lambda_klg
                * kl_factor
            )

            loss = cross_entropy_loss + kld + klg

            # Log things
            self.log("train_kld", kld)
            self.log("train_klg", klg)
            # Nvib parameters
            for layer in range(0, len(model_outputs["latent_dict_list"])):
                self.log(
                    f"train_avg_alpha0_layer_{layer}",
                    model_outputs["latent_dict_list"][layer]["avg_alpha0"],
                )
                self.log(
                    f"train_avg_num_vec_layer_{layer}",
                    model_outputs["latent_dict_list"][layer]["avg_num_vec"],
                )
                self.log(
                    f"train_avg_prop_vec_layer_{layer}",
                    model_outputs["latent_dict_list"][layer]["avg_prop_vec"],
                )

        else:
            loss = cross_entropy_loss
        self.log("train_loss", loss)
        self.log("train_cross_entropy", cross_entropy_loss)

        # Make batch first for plotting
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        # Plot attention
        if self.plot_cross_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(1, 2):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "cross_attentions",
                        self.logger,
                        "Train",
                        zmax=1,
                        # prior="<PRIOR>" if self.is_nvib else None,
                        num_heads=self.model.nhead,
                        num_layers=self.model.num_decoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )
        if self.plot_encoder_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(1, 2):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "encoder_attentions",
                        self.logger,
                        "Train",
                        zmax=1,
                        prior="<PRIOR>" if self.is_nvib else None,
                        num_heads=self.model.nhead,
                        num_layers=self.model.num_encoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        bsz = batch["input_ids"].shape[0]
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        # Forward pass
        model_outputs = self.model(**batch)

        # Get loss and ignore the pads
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="mean")
        # Transform targets
        targets = torch.flatten(batch["labels"].transpose(0, 1))  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(model_outputs["logits"], start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]

        if self.is_nvib:
            kl_factor = self.kl_annealing_scheduler(self.global_step)
            kld = (
                weighted_mean(model_outputs["kl_dirichlet"], self.weighted_kl)
                * self.lambda_kld
                * kl_factor
            )
            klg = (
                weighted_mean(model_outputs["kl_gaussian"], self.weighted_kl)
                * self.lambda_klg
                * kl_factor
            )

            loss = cross_entropy_loss + kld + klg

            # Log things
            self.log("val_kld", kld, batch_size=bsz)
            self.log("val_klg", klg, batch_size=bsz)
            # Nvib parameters
            for layer in range(0, len(model_outputs["latent_dict_list"])):
                self.log(
                    f"val_avg_alpha0_layer_{layer}",
                    model_outputs["latent_dict_list"][layer]["avg_alpha0"],
                    batch_size=bsz,
                )
                self.log(
                    f"val_avg_num_vec_layer_{layer}",
                    model_outputs["latent_dict_list"][layer]["avg_num_vec"],
                    batch_size=bsz,
                )
                self.log(
                    f"val_avg_prop_vec_layer_{layer}",
                    model_outputs["latent_dict_list"][layer]["avg_prop_vec"],
                    batch_size=bsz,
                )
        else:
            loss = cross_entropy_loss

        # Throws a warning if batch size is not specified even though losses are 1D tensors?
        self.log("val_loss", loss, batch_size=bsz)
        self.log("val_cross_entropy", cross_entropy_loss, batch_size=bsz)

        # Autoregressive prediction ()
        generated_ids = self.model.generate(
            batch["input_ids"],
            max_new_tokens=256,
        )
        prompt = self.tokenizer.batch_decode(batch["input_ids"].transpose(0, 1))
        # prompt = strip_after_token(prompt, self.tokenizer.sep_token)
        batch_predictions = self.tokenizer.batch_decode(generated_ids.transpose(0, 1))
        batch_predictions = strip_after_token(batch_predictions, self.tokenizer.sep_token)
        tgt = self.tokenizer.batch_decode(batch["labels"])
        tgt = strip_after_token(tgt, self.tokenizer.sep_token)

        # Make batch first for plotting
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)
        # Plot attention
        if self.plot_cross_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(1, 2):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "cross_attentions",
                        self.logger,
                        "Validation",
                        zmax=1,
                        # prior="<PRIOR>" if self.is_nvib else None,
                        num_heads=self.model.nhead,
                        num_layers=self.model.num_decoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )
        if self.plot_encoder_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(1, 2):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "encoder_attentions",
                        self.logger,
                        "Validation",
                        zmax=1,
                        prior="<PRIOR>" if self.is_nvib else None,
                        num_heads=self.model.nhead,
                        num_layers=self.model.num_encoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )
        return {
            "preds": batch_predictions,
            "targets": tgt,
            "cross_entropy_loss": cross_entropy_loss,
        }

    def validation_epoch_end(self, validation_step_outputs):                                        
        # Calculate score
        preds = []
        targets = []
        cross_entropy_losses = []

        for output in validation_step_outputs:
            preds.append(output["preds"])
            targets.append(output["targets"])
            cross_entropy_losses.append(output["cross_entropy_loss"])

        preds = list(chain.from_iterable(preds))
        targets = list(chain.from_iterable(targets))
        cross_entropy_loss = torch.mean(torch.stack(cross_entropy_losses))

        # Average of all cross entropy losses from teacher forcing
        self.log("ce_val", cross_entropy_loss)

        if self.log_rouge:
            score = evaluate.load("rouge")
            results = score.compute(predictions=preds, references=[[ref] for ref in targets])
            self.log("rouge1_val", results["rouge1"] * 100)
            self.log("rouge2_val", results["rouge2"] * 100)
            self.log("rougeL_val", results["rougeL"] * 100)
        if self.log_bleu:
            score = evaluate.load("sacrebleu")
            results = score.compute(predictions=preds, references=[[ref] for ref in targets])
            self.log("bleu_val", results["score"])
        if self.log_chrf:
            score = evaluate.load("chrf")
            results = score.compute(
                predictions=preds, references=[[ref] for ref in targets], lowercase=True
            )
            self.log("chrf_val", results["score"])

    def test_step(self, batch, batch_idx):
        # Forward pass
        bsz = batch["input_ids"].shape[0]
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)
        model_outputs = self.model(**batch)

        # Get loss and ignore the pads
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="mean")
        # Transform targets
        targets = torch.flatten(batch["labels"].transpose(0, 1))  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(model_outputs["logits"], start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]
        if self.is_nvib:
            kl_factor = self.kl_annealing_scheduler(self.global_step)
            kld = (
                weighted_mean(model_outputs["kl_dirichlet"], self.weighted_kl)
                * self.lambda_kld
                * kl_factor
            )
            klg = (
                weighted_mean(model_outputs["kl_gaussian"], self.weighted_kl)
                * self.lambda_klg
                * kl_factor
            )

            loss = cross_entropy_loss + kld + klg
            # Log things
            self.log("test_kld", kld, batch_size=bsz)
            self.log("test_klg", klg, batch_size=bsz)
        else:
            loss = cross_entropy_loss

        self.log("test_loss", loss, batch_size=bsz)
        self.log("test_cross_entropy", cross_entropy_loss, batch_size=bsz)

        # Autoregressive prediction
        generated_ids = self.model.generate(
            batch["input_ids"],
            max_new_tokens=256,
        )
        batch_predictions = self.tokenizer.batch_decode(generated_ids.transpose(0, 1))
        batch_predictions = strip_after_token(batch_predictions, self.tokenizer.sep_token)
        tgt = self.tokenizer.batch_decode(batch["labels"])
        tgt = strip_after_token(tgt, self.tokenizer.sep_token)

        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        # Plot attention
        if self.plot_cross_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(0, 3):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "cross_attentions",
                        self.logger,
                        "Test",
                        zmax=1,
                        # prior="<PRIOR>" if self.is_nvib else None,
                        num_heads=self.model.nhead,
                        num_layers=self.model.num_decoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )
        if self.plot_encoder_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(0, 3):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "encoder_attentions",
                        self.logger,
                        "Test",
                        zmax=1,
                        prior="<PRIOR>" if self.is_nvib else None,
                        num_heads=self.model.nhead,
                        num_layers=self.model.num_encoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )

        return {
            "preds": batch_predictions,
            "targets": tgt,
            "cross_entropy_loss": cross_entropy_loss,
        }

    def test_epoch_end(self, validation_step_outputs):
        # Calculate score
        preds = []
        targets = []
        cross_entropy_losses = []

        for output in validation_step_outputs:
            preds.append(output["preds"])
            targets.append(output["targets"])
            cross_entropy_losses.append(output["cross_entropy_loss"])

        preds = list(chain.from_iterable(preds))
        targets = list(chain.from_iterable(targets))
        cross_entropy_loss = torch.mean(torch.stack(cross_entropy_losses))

        # Average of all cross entropy losses from teacher forcing
        self.log("ce_test", cross_entropy_loss)

        if self.log_rouge:
            score = evaluate.load("rouge")
            results = score.compute(predictions=preds, references=[[ref] for ref in targets])
            self.log("rouge1_test", results["rouge1"] * 100)
            self.log("rouge2_test", results["rouge2"] * 100)
            self.log("rougeL_test", results["rougeL"] * 100)
        if self.log_bleu:
            score = evaluate.load("sacrebleu")
            results = score.compute(predictions=preds, references=[[ref] for ref in targets])
            self.log("bleu_test", results["score"])
        if self.log_chrf:
            score = evaluate.load("chrf")
            results = score.compute(
                predictions=preds, references=[[ref] for ref in targets], lowercase=True
            )
            self.log("chrf_test", results["score"])

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)

        if self.learning_rate_scheduler:
            lr_scheduler = {
                # "scheduler": OneCycleLR(
                #     optimizer,
                #     max_lr=self.learning_rate,
                #     total_steps=self.trainer.max_steps,
                #     pct_start=0.1,
                #     anneal_strategy="linear",
                # ),
                "scheduler": get_cosine_schedule_with_warmup(
                    optimizer,
                    num_training_steps=self.trainer.max_steps,
                    num_warmup_steps=self.perc_warmup * self.trainer.max_steps,
                ),
                "name": "learning_rate_scheduler",
                "interval": "step",
            }

            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
