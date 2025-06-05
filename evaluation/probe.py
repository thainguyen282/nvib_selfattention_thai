#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#


import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class AttentionBasedClassifier(nn.Module):
    def __init__(
            self,
            dim: object,
            n_output: object,
            hidden_dim=64,
            aggregation=None,
            use_pi=False,
    ) -> object:
        super(AttentionBasedClassifier, self).__init__()
        self.use_pi = use_pi

        self.vec = nn.Parameter(torch.randn(1, dim))  # learnable query vector

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.scale = dim ** -0.5
        self.eps = 1e-8
        self.lin = nn.Linear(dim, n_output)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.ReLU(inplace=True), nn.Linear(2 * dim, n_output)
        )
        self.inp_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, representations, mask, pi=None):
        b, n, dim = representations.shape

        representations = self.inp_mlp(representations)

        vec = self.vec.expand(b, -1)

        k, v = self.to_k(representations), self.to_v(representations)

        q = self.to_q(vec)

        dots = torch.einsum("bd,bjd->bj", q, k) * self.scale

        dots.masked_fill_(mask, float("-inf"))

        if self.use_pi and pi is not None:
            dots.masked_fill_((pi == 0.0), float("-inf"))

        attn = dots.softmax(dim=-1) + self.eps

        updates = torch.einsum("bjd,bj->bd", v, attn)

        return self.lin(updates), attn


class AggregatingProbe(nn.Module):
    def __init__(
            self, dim, n_output, hidden_dim=256, aggregation=torch.mean, use_pi=False
    ):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_output),
        )
        self.lin = nn.Linear(dim, n_output)
        self.aggregate = aggregation
        self.use_pi = use_pi

    def forward(self, representations, mask, pi=None):
        _mask = mask.unsqueeze(2).repeat(1, 1, self.dim)
        if self.aggregate == torch.max:
            representations.masked_fill_(_mask, float("-inf"))
            aggregated_representation = self.aggregate(representations, dim=-2).values
        elif self.aggregate == torch.sum:
            representations.masked_fill_(_mask, 0.0)
            if self.use_pi and pi is not None:
                # weighting by pi s then the output would be a weighted mean
                representations = representations * pi.repeat(1, 1, self.dim)
            aggregated_representation = self.aggregate(representations, dim=-2)
        elif self.aggregate == torch.mean:
            representations.masked_fill_(_mask, 0.0)
            if self.use_pi and pi is not None:
                representations = representations.masked_fill_(
                    (pi == 0.0), 0.0
                )  # representations * pi.repeat(1, 1, self.dim)
                # MB: we did not have the denumerator in the reported results
                aggregated_representation = torch.sum(
                    representations, dim=-2
                ) / torch.sum(
                    (mask * (pi == 0)).to(float) == 0.0, dim=-1, keepdim=True
                ).repeat(
                    1, self.dim
                )
            else:
                aggregated_representation = torch.sum(
                    representations, dim=-2
                ) / torch.sum(mask.to(float) == 0.0, dim=-1, keepdim=True).repeat(
                    1, self.dim
                )

        # breakpoint()
        return self.mlp(aggregated_representation), None


class LitProbe(pl.LightningModule):
    def __init__(self, model, probe, args, num_classes=2):
        super().__init__()
        self.model = model
        self.probe = probe
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.lr = args.learning_rate
        self.metric = (
            MulticlassAccuracy(num_classes=num_classes)
            if args.data == "senteval"
            else MulticlassF1Score(num_classes=num_classes)
        )
        self.args = args

    def training_step(self, batch, batch_idx):
        self.model.eval()
        bsz = batch["input_ids"].shape[0]
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["labels"] = batch["labels"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        # Forward pass
        model_output = self.model.model(**batch)

        representation = self._get_representation(model_output)
        mask = self._get_mask(model_output)
        pi = self._get_pi(model_output)
        pi = pi.transpose(0, 1) if pi is not None else pi
        probe_out, probe_attn = self.probe(representation.transpose(0, 1), mask, pi)
        loss = self.criterion(probe_out, batch["class"])
        # breakpoint()
        acc = self.metric(
            probe_out, batch["class"]
        )  # ((torch.argmax(probe_out, dim=-1) == batch["class"]).float()).mean()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=bsz
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, logger=True, batch_size=bsz
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        bsz = batch["input_ids"].shape[0]
        batch["input_ids"] = batch["input_ids"].transpose(0, 1)
        batch["labels"] = batch["labels"].transpose(0, 1)
        batch["decoder_input_ids"] = batch["decoder_input_ids"].transpose(0, 1)

        # Forward pass
        self.model.eval()
        model_output = self.model.model(**batch)

        representation = self._get_representation(model_output)
        mask = self._get_mask(model_output)
        pi = self._get_pi(model_output)
        pi = pi.transpose(0, 1) if pi is not None else pi
        output, _ = self.probe(representation.transpose(0, 1), mask, pi)
        target = batch["class"]
        loss = self.criterion(output, target)
        acc = self.metric(
            output, target
        )  # torch.sum(torch.argmax(output, dim=-1) == torch.tensor(batch["class"]).float()).item()
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=bsz,
        )
        self.log(
            f"{prefix}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=bsz,
        )

        return {"outputs": output, "targets": target}

    def _get_representation(self, output):
        layer_num = self.args.probe_layer_num

        if self.args.model == "Transformer":
            return output["outputs"][layer_num]
        else:
            # this part is compatible only with a 6 layer Transformer with 3 NVIB layers. Change if necessary.
            if 0 > layer_num >= -3 or layer_num > 2:
                return output["latent_dict_list"][layer_num]["z"][0]
            else:
                if layer_num < 0:
                    layer_num += 6
                return output["old_memory"][layer_num]

    def _get_mask(self, output):
        if self.args.model == "Transformer":
            return output["memory_key_padding_mask"]
        else:
            # this part is compatible only with a 6 layer Transformer with 3 NVIB layers. Change if necessary.
            if 0 > self.args.probe_layer_num > -3 or self.args.probe_layer_num > 2:
                return output["latent_dict_list"][self.args.probe_layer_num][
                    "memory_key_padding_mask"
                ]
            else:
                return output["old_memory_mask"]

    def _get_pi(self, output):
        if self.args.model == "Transformer":
            return None
        else:
            return output["latent_dict_list"][self.args.probe_layer_num]["pi"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.probe.parameters(), lr=self.lr)
