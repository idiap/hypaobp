#
# SPDX-FileCopyrightText: 2020 Idiap Research Institute <contact@idiap.ch>
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#            Florian Mai <florian.mai@idiap.ch>
#            Thijs Vogels <thijs.vogels@epfl.ch>
#
# SPDX-License-Identifier: MIT
#

import json
import os
import re
from glob import glob

import pandas as pd
import numpy as np
import argparse


def generate_dataframe(inpath):
    subdirectories = [os.path.join(inpath, o) for o in os.listdir(inpath) if os.path.isdir(os.path.join(inpath,o))]
    optimizer_labels = [o for o in os.listdir(inpath) if os.path.isdir(os.path.join(inpath,o))]

    result_list = []
    for o_label, optim_directory in zip(optimizer_labels, subdirectories):
        for path in glob(optim_directory + "/**/*.json", recursive=True):
            result_list.append(read_logfile(path, o_label))

    df = pd.DataFrame(result_list)
    return df


#def derive_optimizer_name(path):
#    for optimizer in sorted(
#        ["Adagrad", "Adam", "AdamLR", "SGD", "SGDM", "SGDMC", "SGDMW", "SGDMCWC", "SGDDecay"],
#        key=lambda v: -len(v),
#    ):
#        if optimizer in path or optimizer.lower() in path:
#            return optimizer
#
#    raise Exception(f"Couldn't determine optimizer name in path {path}")


def read_logfile(path, optimizer_label):
    with open(path) as fp:
        data = json.load(fp)
    return {
        "problem": data["testproblem"],
        "optimizer": optimizer_label,
        "seed": data["random_seed"],
        "batch_size": data["batch_size"],
        "path": path,
        "num_epochs": data["num_epochs"],
        "weight_decay": data["weight_decay"],
        **data["optimizer_hyperparams"],
        "mean_wall_clock_time": np.mean(data["wall_clock_time"]),
        "num_steps": len(data["wall_clock_time"]),
        "best_test_loss": min(data["test_losses"]) if "test_losses" in data else pd.NaT,
        "best_valid_loss": min(data["valid_losses"]) if "valid_losses" in data else pd.NaT,
        "best_train_loss": min(data["train_losses"]) if "train_losses" in data else pd.NaT,
        "best_test_accuracy": max(data["test_accuracies"]) if "test_accuracies" in data else pd.NaT,
        "best_valid_accuracy": max(data["valid_accuracies"])
        if "valid_accuracies" in data
        else pd.NaT,
        "best_train_accuracy": max(data["train_accuracies"])
        if "train_accuracies" in data
        else pd.NaT,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Write experiment results into a dataframe that is more convenient to work with.')
    parser.add_argument('-inpath', type=str, required=True,
                        help='Path to results.')
    parser.add_argument('-outfile', type=str, default="results.pickle",
                        help='Where to store the results dataframe.')

    args = parser.parse_args()

    df = generate_dataframe(args.inpath)
    outfile = args.outfile
    df.to_pickle(outfile)
    print("Saved to {}".format(os.path.realpath(outfile)))

