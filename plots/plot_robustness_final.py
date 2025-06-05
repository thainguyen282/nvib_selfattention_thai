#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    df = pd.DataFrame(
        {
            "Model": [
                "NVIB",
                "NVIB",
                "NVIB",
                "NVIB",
                "NVIB",
                "NVIB",
                "Transformer",
                "Transformer",
                "Transformer",
                "Transformer",
                "Transformer",
                "Transformer",
            ],
            "Cross-Entropy": [
                0,
                0.15157,
                0.30017,
                0.40957,
                0.43117,
                0.42647,
                0,
                0.23954,
                0.45814,
                0.62794,
                0.66734,
                0.66714,
            ],
            "Self-ChrF": [
                0,
                -7.745,
                -15.668,
                -21.483,
                -22.639,
                -22.822,
                0,
                -9.986,
                -20.522,
                -28.429,
                -30.108,
                -30.246,
            ],
            "Probability": [0, 0.2, 0.4, 0.6, 0.8, 1, 0, 0.2, 0.4, 0.6, 0.8, 1],
        }
    )
    print(df)

    colors = [
        "#8A2BE2",
        "#277f8e",
    ]
    # set marker shapes
    new_markers = ["o", "X"]

    # Set your custom color palette
    customPalette = sns.set_palette(sns.color_palette(colors))
    sns.lineplot(
        data=df,
        x="Probability",
        y="Cross-Entropy",
        hue="Model",
        style="Model",
        palette=customPalette,
        linewidth=2,
        markersize=10,
        markers=new_markers,
        alpha=0.7,
    )
    fontsize = 18

    plt.xlabel("Probability of noise", fontsize=fontsize)
    plt.ylabel(r"$\Delta$ Cross-Entropy", fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.legend(markerscale=1.5, fontsize=fontsize)
    # plt.show()

    # Save as pdf
    plt.savefig("Robustness_ce.pdf", bbox_inches="tight")
    plt.close()

    sns.lineplot(
        data=df,
        x="Probability",
        y="Self-ChrF",
        # markers=True,
        hue="Model",
        style="Model",
        palette=customPalette,
        linewidth=2,
        markersize=10,
        markers=new_markers,
        alpha=0.7,
    )

    plt.xlabel("Probability of noise", fontsize=fontsize)
    plt.ylabel(r"$\Delta$ Self-ChrF", fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.legend(markerscale=1.5, fontsize=fontsize)
    # plt.show()

    # Save as pdf
    plt.savefig("Robustness_chrf.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
