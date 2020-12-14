#
# SPDX-FileCopyrightText: 2020 Idiap Research Institute <contact@idiap.ch>
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#            Florian Mai <florian.mai@idiap.ch>
#            Thijs Vogels <thijs.vogels@epfl.ch>
#
# SPDX-License-Identifier: MIT
#


import itertools
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import seaborn as sns
import argparse

ACCURACY_PROBLEMS = set(
    [
        "fmnist_2c2d",
        "cifar100_allcnnc",
        "cifar10_3c3d",
        "imdb_bilstm",
        "svhn_wrn164",
        "tolstoi_char_rnn",
    ]
)
LOSS_PROBLEMS = set(["mnist_vae", "quadratic_deep", "fmnist_vae"])


def load_dataframe():
    if os.path.isfile("results.pickle"):
        return pd.read_pickle("results.pickle")
    else:
        from results_create_dataframe import generate_dataframe, problems

        return generate_dataframe(problems)


def expected_quality_with_budget(subset, problem, optimizer, budget, num_shuffles=1000):
    if problem in ACCURACY_PROBLEMS:
        losses = subset.best_test_accuracy
        results = []
        for i in range(num_shuffles):
            results.append(losses.sample(budget).max())
        return np.mean(results), np.std(results)
    elif problem in LOSS_PROBLEMS:
        losses = subset.best_test_loss
        results = []
        for i in range(num_shuffles):
            results.append(losses.sample(budget).min())
        return np.mean(results), np.std(results)


def best_with_budget(dataset, problem, optimizers, budget, num_shuffles=1000):
    losses_for_optimizer = {}
    for optimizer in optimizers:
        subset = dataset[dataset.problem.eq(
            problem) & dataset.optimizer.eq(optimizer)]
        losses_for_optimizer[optimizer] = (
            subset.best_test_accuracy if problem in ACCURACY_PROBLEMS else subset.best_test_loss
        )

    if problem in ACCURACY_PROBLEMS:
        res = {optimizer: 0.0 for optimizer in optimizers}
        for i in range(num_shuffles):
            best_so_far = ("", -np.inf)
            for optimizer in optimizers:
                val = losses_for_optimizer[optimizer].sample(budget).max()
                if val > best_so_far[1]:
                    best_so_far = (optimizer, val)
            best_optimizer, _ = best_so_far
            res[best_optimizer] += 1.0 / num_shuffles
        return res
    elif problem in LOSS_PROBLEMS:
        res = {optimizer: 0.0 for optimizer in optimizers}
        for i in range(num_shuffles):
            best_so_far = ("", np.inf)
            for optimizer in optimizers:
                val = losses_for_optimizer[optimizer].sample(budget).min()
                if val < best_so_far[1]:
                    best_so_far = (optimizer, val)
            best_optimizer, _ = best_so_far
            res[best_optimizer] += 1.0 / num_shuffles
        return res


def compute_quality_for_budget(x):
    subset, problem, optimizer, budget = x
    mean, std = expected_quality_with_budget(subset, problem, optimizer, budget)
    return {
        "problem": problem,
        "budget": budget,
        "optimizer": optimizer,
        "L_mean": mean,
        "L_std": std,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computes the winning optimizer from a given dataframe and stores the results into a new dataframe.')
    parser.add_argument('-inpath', type=str, required=True,
                        help='Path to results.')
    parser.add_argument('-budget', type=int, required=True,
                        help='The budget each optimizer has.')
    parser.add_argument('-stepsize', type=int, default=2,
                        help='The resolution of the x axis (budget).')
    parser.add_argument('-outfile_best_budget', default = "results_best_with_budget.pickle", help="Path to save the output file including the best optimizer at each budget to.")
    parser.add_argument('-outfile', default = "results_processed.pickle", help="Path to save the output file to.")

    args = parser.parse_args()

    pool = mp.Pool(8)
    data = pd.read_pickle(args.inpath)

    problems = data.problem.unique()
    optimizers = data.optimizer.unique()

    table = []

    best_with_budget_table = []
    for problem in problems:
        for budget in range(1, args.budget, args.stepsize):
            best_with_budget_table.append(
                {
                    **best_with_budget(data, problem, optimizers, budget),
                    "budget": budget,
                    "problem": problem,
                }
            )

        for optimizer in optimizers:
            print(problem, optimizer)
            subset = data[data.problem.eq(
                problem) & data.optimizer.eq(optimizer)].copy()
            print(len(subset))
            table.extend(pool.map(compute_quality_for_budget, list(map(lambda e: (subset, problem, optimizer, e), range(1, args.budget, args.stepsize)))))
            # for budget in range(1, 65):
            #     mean, std = expected_quality_with_budget(
            #         subset, problem, optimizer, budget)
            #     table.append(
            #         {
            #             "problem": problem,
            #             "budget": budget,
            #             "optimizer": optimizer,
            #             "L_mean": mean,
            #             "L_std": std,
            #         }
            #     )

    derived_df = pd.DataFrame(table)

    derived_df.to_pickle(args.outfile)

    pd.DataFrame(best_with_budget_table).to_pickle(
        args.outfile_best_budget)
