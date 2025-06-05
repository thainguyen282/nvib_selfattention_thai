#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import glob
import os
import random

import numpy as np
import plotly.express as px
import torch

import wandb


def reproducibility(SEED):
    """
    Set all seeds
    :param SEED: int
    :return:
    """
    # Reproducability seeds + device
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        DEVICE = torch.device("cpu")

    # Generator for dataloader
    g = torch.Generator()
    g.manual_seed(SEED)

    return DEVICE, g


def strip_after_token(sents, token):
    """
    Strip the all tokens after and including the specified token from list of lists
    eg [['[CLS] the [UNK] peninsula [SEP] [PAD] the cat [PAD] [PAD]']] -> [['[CLS] the [UNK] peninsula']]
    :param sents: list of lists of sentences
    :param tokenizer: tokenizer
    :return: stripped sentences
    """
    return [sent[: max((sent.index(token)), 0)] if token in sent else sent for sent in sents]


def get_checkpoint_path(path):
    ckpt_lst = glob.glob(os.path.join(path, "epoch*step*.ckpt"))
    if len(ckpt_lst) != 0:
        checkpoint_path = ckpt_lst[0]
    else:
        checkpoint_path = None

    return checkpoint_path


def get_best_model_path(path):
    ckpt_lst = glob.glob(os.path.join(path, "best_model.ckpt"))
    if len(ckpt_lst) != 0:
        checkpoint_path = ckpt_lst[0]
    else:
        checkpoint_path = None

    return checkpoint_path


def load_model(checkpoint_path, model_name, args):
    print("Loading model: ", checkpoint_path)
    model = model_name.load_from_checkpoint(checkpoint_path, args=args, strict=False)
    return model

def load_model_strictly(checkpoint_path, model_name, args):
    print("Loading model: ", checkpoint_path)
    model = model_name.load_from_checkpoint(checkpoint_path)
    return model

def create_or_load_model(output_path, checkpoint_path, model_name, args):
    # Create model and load from checkpoint
    if checkpoint_path is not None:
        print("Loading model")
        model = load_model(checkpoint_path, model_name, args=args)
    else:
        print("Creating model")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        model = model_name(args)

    if os.path.exists(os.path.join(output_path, "wandb_id.txt")):
        print("Loading W&B ID")
        wandb_id = open(os.path.join(output_path, "wandb_id.txt"), "r").read()
    else:
        print("Creating W&B ID")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(output_path, "wandb_id.txt"), "w") as f:
            f.write(wandb_id)

    return model, wandb_id


def show_attention(
    batch,
    batch_item,
    tokenizer,
    attentions_all,
    attention_type,
    logger,
    val_or_test,
    zmax=None,
    prior=None,
    num_heads=1,
    num_layers=1,
    batch_idx=None,
    observation=None,
):
    # num_layers = len(model_outputs["cross_attentions"])
    # num_heads = model_outputs["cross_attentions"][0].shape[1]

    attentions = attentions_all[attention_type]

    if attention_type == "cross_attentions":
        input_sentence = tokenizer.convert_ids_to_tokens(batch["input_ids"][batch_item, :])
        input_sentence = strip_after_token([input_sentence], tokenizer.sep_token)[0] + ["<EOS>"]
        if prior is not None:
            input_sentence = [prior] + input_sentence
        output_ids = batch["decoder_input_ids"][batch_item, :]

    else:
        # Self attention or encoder attention
        input_sentence = tokenizer.convert_ids_to_tokens(batch["input_ids"][batch_item, :])
        input_sentence = strip_after_token([input_sentence], tokenizer.sep_token)[0] + ["<EOS>"]
        if prior is not None:
            input_sentence = [prior] + input_sentence
        output_ids = batch["input_ids"][batch_item, :]

    output_sentence = tokenizer.convert_ids_to_tokens(output_ids)
    output_sentence = strip_after_token([output_sentence], tokenizer.sep_token)[0]

    # Make the tokens unique by adding a count
    count = 1
    for i in range(0, len(input_sentence)):
        token = input_sentence[i]
        input_sentence[i] = token + "_" + str(count)
        count += 1

    count = 1
    for i in range(0, len(output_sentence)):
        token = output_sentence[i]
        output_sentence[i] = token + "_" + str(count)
        count += 1

    # Plot layers and heads
    num_layers = len(attentions) # HACK to plot all layers  
    for layer in range(num_layers):
        for head in range(num_heads):
            if num_heads == 1:
                cross_attentions = attentions[layer][batch_item, :, :, :].mean(dim=0)
            else:
                cross_attentions = attentions[layer][batch_item, head, :, :]
            fig = px.imshow(
                cross_attentions[: len(output_sentence), : len(input_sentence)].detach().cpu(),
                labels=dict(x="K", y="Q", color="Score"),
                zmax=zmax,
                zmin=0,
                x=input_sentence,
                y=output_sentence,
            )
            # Access the wandb logger to log a figure
            if num_heads == 1:
                logger.experiment.log(
                    {
                        f"{val_or_test} - {attention_type} Map - Layer {layer} - Pooled Heads - Batch {batch_idx} - Obs {observation}": wandb.Plotly(
                            fig
                        )
                    }
                )
            else:
                logger.experiment.log(
                    {
                        f"{val_or_test} - Attention Map - Layer {layer} - Head {head} - Batch {batch_idx} - Obs {observation}": wandb.Plotly(
                            fig
                        )
                    }
                )
