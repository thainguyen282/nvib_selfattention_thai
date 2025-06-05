#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    df = pd.DataFrame(
        {
            "Task": [
                "BShift",
                "BShift",
                "BShift",
                "CoordInv",
                "CoordInv",
                "CoordInv",
                # "ObjNum",
                # "ObjNum",
                # "ObjNum",
                "Tense",
                "Tense",
                "Tense",
                "TopConst",
                "TopConst",
                "TopConst",
                # "TreeDepth",
                # "TreeDepth",
                # "TreeDepth",
            ],
            "Accuracy": [
                4.953014879,
                5.033936652,
                11.16193962,
                -1.655119323,
                1.982294072,
                5.662904152,
                # 2.497723429,
                # -0.9952177847,
                # 5.814853017,
                4.523046586,
                0.6129807692,
                1.753339695,
                21.10805861,
                20.92011902,
                36.89458689,
                # -1.374442793,
                # -3.106725146,
                # 6.484517304,
            ],
            "Layers": [
                4,
                5,
                6,
                4,
                5,
                6,
                4,
                5,
                6,
                4,
                5,
                6,
                #    4, 5, 6, 4, 5, 6
            ],
        }
    )
    print(df)

    colors = [
        "#4E78A0",
        "#4ac16d",
        "#8A2BE2",
        "#FFB81C",
    ]
    # set marker shapes
    new_markers = ["o", "v", "X", "P", "X", "D", "p", "x", "d"]

    # Set your custom color palette
    customPalette = sns.set_palette(sns.color_palette(colors))
    sns.lineplot(
        data=df,
        x="Layers",
        y="Accuracy",
        # markers=True,
        hue="Task",
        style="Task",
        palette=customPalette,
        linewidth=2,
        markersize=10,
        markers=new_markers,
        alpha=0.7,
    )

    fontsize = 15
    plt.xticks(np.arange(4, 7), ("4", "5", "6"))
    plt.ylim(-5, 40)
    plt.xlabel("Layer", fontsize=fontsize)
    plt.ylabel("Relative task accuracy [%]", fontsize=fontsize)
    plt.axhline(y=0, color="black", linestyle="--")
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.legend(markerscale=1.5, fontsize=fontsize)
    # plt.show()

    # Save as pdf
    plt.savefig("senteval.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
