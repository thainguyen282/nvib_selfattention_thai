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

import pandas as pd

from utils import *


def main(args):
    df = pd.DataFrame(columns=["model_name", "deletion_prob", "bleu_score"])
    for model_name in args.model_list:
        print("Model name: ", model_name)
        OUTPUT_PATH = os.path.join(
            args.output_dir,
            args.project_name,
            model_name,
        )
        label = (
            "wdtd"
            if args.token_deletion and args.word_deletion
            else "wd"
            if args.word_deletion
            else "td"
        )
        robustness_df = pd.read_csv(
            os.path.join(OUTPUT_PATH, f"bleu_scores_{label}_deletion_probs.csv"),
            index_col=0,
        )
        # add model name before underscore
        model_name = model_name.split("_")[0]
        robustness_df["model_name"] = model_name
        # append to df
        df = pd.concat((df, robustness_df), ignore_index=True)

    # plot bleu score vs deletion prob together
    fig = px.line(
        df,
        x="deletion_prob",
        y="bleu_score",
        color="model_name",
        title=f"BLEU score vs deletion {label} probability",
    )
    fig.show()
    # Save figure
    fig.write_image(
        os.path.join(
            args.output_dir,
            args.project_name,
            f"bleu_score_vs_{label}_deletion_prob.pdf",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_list", nargs="+", default=[])
    parser.add_argument("-o", "--output_dir", default="outputs")
    parser.add_argument("-p", "--project_name", default="robustness")
    parser.add_argument("--token_deletion", action="store_true", help="Delete tokens")
    parser.add_argument("--word_deletion", action="store_true", help="Delete words")

    args = parser.parse_args()
    main(args)

# python plots/plot_robustness.py --project_name baseline_evaluation --model_list NVIBSaTransformer_elyr3_klg0.01_kld1_delta0.1__ms8000/ Transformer_ms8000/
