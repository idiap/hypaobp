#
# SPDX-FileCopyrightText: 2020 Idiap Research Institute <contact@idiap.ch>
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#            Florian Mai <florian.mai@idiap.ch>
#            Thijs Vogels <thijs.vogels@epfl.ch>
#
# SPDX-License-Identifier: MIT
#

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import argparse

from results_process import ACCURACY_PROBLEMS, LOSS_PROBLEMS

sns.set_style("whitegrid")
# matplotlib.rcParams["text.usetex"] = True
colors = ['palegreen', 'limegreen', 'darkgreen', 'lightcoral', 'brown', 'red', 'darkred', 'tomato']
# labels=["Adagrad", "Adam", "AdamLR", "SGD", "SGDM", "SGDMC", "SGDMW", "SGDMCWC"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates a plot that shows the probability of each optimizer to yield the best performance.')
    parser.add_argument('-inpath', type=str, required=True,
                        help='Path to dataframe that contains the best optimizers for each budget.')
    parser.add_argument('-optimizers', type=str, nargs = "+", required=True, help = "List of optimizers to plot.")
    parser.add_argument('-optimizer_labels', type=str, nargs = "+", required=False, default = None, help = "List of labels to display for each optimizer given in 'optimizers'. If None, the optimizer names are taken directly as labels.")
    parser.add_argument('-budget', required=True, type=int, help="Maximum budget to consider.")
    parser.add_argument('-usetex', action="store_true", help="Use tex for printing labels.")
    parser.add_argument('-global_averaging', action="store_true", help="Average over all problems.")
    parser.add_argument('-outfile', type=str, default = "best_optimizer_avg.pdf", help="Path to store the plot.")
    args = parser.parse_args()

    matplotlib.rcParams['text.usetex'] = args.usetex
    global_averaging = args.global_averaging

    if args.optimizer_labels is None:
        args.optimizer_labels = args.optimizers

    df = pd.read_pickle(args.inpath)
    df["problemlabel"] = df.problem.replace(
        {
            "fmnist_2c2d": r"F-MNIST 2C2D",
            "cifar100_allcnnc": r"CIFAR 100",
            "cifar10_3c3d": r"CIFAR 10",
            "imdb_bilstm": r"IMDB LSTM",
            "svhn_wrn164": r"SVHN WRN164",
            "tolstoi_char_rnn": r"Tolstoi Char-RNN",
            "mnist_vae": r"MNIST VAE",
            "quadratic_deep": r"Quadratic Deep",
            "fmnist_vae": r"F-MNIST VAE",
        }
    )

    df["Budget"] = df.budget

    def plot(budget, *arguments, **kwargs):
        ax = plt.gca()
        ax.stackplot(
            budget,
            *[a.rolling(1, win_type="triang").mean() for a in arguments],
            labels=args.optimizer_labels,
            colors=colors
        )

    if not global_averaging:
        grid = sns.FacetGrid(
            data=df,
            col="problemlabel",
            col_wrap=3,
            sharey=True,
            size=2.3,
            aspect=1.61,
            gridspec_kws={"wspace": 0.1, "hspace": 0.1},
        )

        grid.map(
            plot, "Budget", *args.optimizers
        )
        grid.set_titles("{col_name}", "", "")
        plt.tight_layout(0.5)
        i = 0
        for ax in grid.axes.flat:
            if i % 3 == 0:
                ax.set_ylabel("Prob. of finding best optimizer")
            i += 1
    else:
        df = df.groupby('budget').agg('mean')
        print(df.columns)
        plt.stackplot(
            df.index,
            df[args.optimizers].T, 
            labels=args.optimizer_labels, 
            colors=colors, 
        )
        plt.xlim((1, args.budget - 1))
        plt.ylim((0, 1))
        plt.xlabel('Number of hyperparameter configurations tried')
        plt.ylabel('\% chance of finding the best configuration')
        plt.tight_layout(0.5)

    plt.legend()
    plt.savefig(args.outfile)
    # plt.show()
