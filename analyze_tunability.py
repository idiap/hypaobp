#
# SPDX-FileCopyrightText: 2020 Idiap Research Institute <contact@idiap.ch>
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#            Florian Mai <florian.mai@idiap.ch>
#            Thijs Vogels <thijs.vogels@epfl.ch>
#
# SPDX-License-Identifier: MIT
#


import multiprocessing as mp
import os
import random
import sys
import argparse

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

sys.path.append("./DeepOBS/")

from deepobs import analyzer
from deepobs.analyzer import (get_hyperparameter_optimization_performance,
                              plot_box_hyperparam_optim,
                              plot_hyperparam_optimization,
                              print_tunability_to_latex)

def temp(x):
    return get_hyperparameter_optimization_performance(x[0], num_shuffle=x[1])

def main():

    parser = argparse.ArgumentParser(description='Computes summary metrics and boxplots for any given problem.')
    parser.add_argument('-inpath', type=str, required=True,
                        help='Basepath to all the logs.')
    parser.add_argument('-problem', type=str, required=True, choices = ["fmnist_2c2d", "mnist_vae", "quadratic_deep", "fmnist_vae", "cifar100_allcnnc", "cifar10_3c3d", "imdb_bilstm", "svhn_wrn164", "tolstoi_char_rnn"], help = "Problem to plot the boxplot for.")
    parser.add_argument('-optimizers', type=str, nargs = "+", required=True, help = "List of optimizers to plot.")
    parser.add_argument('-optimizer_labels', type=str, nargs = "+", required=False, default = None, help = "List of labels to display for each optimizer given in 'optimizers'. If None, the optimizer names are taken directly as labels.")
    parser.add_argument('-num_shuffle', default=100, type=int, help="Number of times to shuffle for computing expected validation performance via bootstrapping.")
    parser.add_argument('-outfile', type=str, default = "boxplot.pdf", help="Path to store the plot.")
    parser.add_argument('-x_axis', type=str, default = "trials", choices = ["trials", "wct"], help="What to consider as budget, either 'trials' (number of hyperparameter configurations) or 'wct' (wallclock-time).")
    parser.add_argument('-print_metrics', action="store_true", help="If set, print metrics (CPE, CPU, etc...) instead of plotting boxplots.")
    args = parser.parse_args()

    random.seed(a=12) # fix random seed for when shuffling the random search results.
    num_shuffle = args.num_shuffle
    mpl.use('pgf')


    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots 
        "font.sans-serif": [],              # to inherit fonts from the document
        "font.monospace": [],
        "font.size": 10,
        "legend.fontsize": 10,               # Make the legend/label fonts 
        "xtick.labelsize": 20,               # a little smaller
        "ytick.labelsize": 10,                              # default fig size of 0.9 textwidth
        "axes.labelsize": 28,
        "pgf.preamble": [
            r"\usepackage[utf8]{inputenc}",    # use utf8 input and T1 fonts 
            r"\usepackage[T1]{fontenc}",  # plots will be generated 
            ]                                   # using this preamble
        }
    #mpl.use("agg")
    mpl.rcParams.update(pgf_with_latex)

    included_metrics = ["Cumulative Performance Early", "Cumulative Performance Uniform", "Cumulative Performance Late", "Sharpness", "Avg WCT", "k=0.1", "k=0.01", "k=0.001", "k=0.0001"]

    problems = [args.problem]

    x_axis = args.x_axis

    axis_limits = {
        'fmnist_2c2d': (80, 94),
        'mnist_vae': (25, 100),
        'quadratic_deep': (80, 150), 
        'fmnist_vae': (20, 80),
        'cifar100_allcnnc': (1, 50),
        'cifar10_3c3d': (40, 90),
        'imdb_bilstm': (60, 90),
        'svhn_wrn164': (80, 99),
        'tolstoi_char_rnn': (10, 70)
    }


    plot_labels = {
        'fmnist_2c2d': 'FMNIST Classification',
        'mnist_vae': 'MNIST VAE',
        'quadratic_deep': 'Quadratic Deep', 
        'fmnist_vae': 'FMNIST VAE',
        'cifar100_allcnnc': 'CIFAR 100',
        'cifar10_3c3d': 'CIFAR 10',
        'imdb_bilstm': 'IMDb LSTM',
        'svhn_wrn164': 'WRN 16(4)',
        'tolstoi_char_rnn': 'Char RNN'
    }

    pool = mp.Pool(16)

    all_axes = []
    all_figs = []
    for prob_no, problem in enumerate(problems):
        print(f"Doing {problem}")
        problem_type = ['acc', 'loss'][('vae' in problem) or ('deep' in problem)]
        root_path = './'

        prob_dir = args.inpath
        dir_path = []
        for opt in args.optimizers:
            path_to_jsons = find_json_parent(os.path.join(prob_dir, opt))
            dir_path.append([path_to_jsons])

        opt_labels = args.optimizer_labels or args.optimizers
        if args.print_metrics:
            hyperparam_perf = analyzer.compute_tunability_metrics(dir_path, opt_labels, obj="max" if problem_type =="acc" else "min", num_shuffle=num_shuffle, x_axis=x_axis)
            print_tunability_to_latex(hyperparam_perf, included_metrics, score_type = problem_type)
        else:        
            all_opts_all_logs = pool.map(temp, zip(dir_path, [num_shuffle] * len(dir_path)))
            fig, ax = plot_box_hyperparam_optim(all_opts_all_logs, labels=opt_labels, x_type='num-evals', score_type = problem_type, y_limits=axis_limits[problem], do_best=False, plot_box=True, do_legend=False)
            plt.suptitle(f'{plot_labels[problem]}', fontsize=15, fontweight='bold')

            plt.savefig(args.outfile)

def find_json_parent(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for d in dirs:
            d = os.path.join(subdir, d)
            for f in os.listdir(d):
                is_correct_dir = f.endswith(".json")
                if is_correct_dir:
                    return subdir
    
if __name__ == "__main__":
    main()
