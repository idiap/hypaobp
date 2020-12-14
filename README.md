# Code for [*Optimizer Benchmarking Needs to Account for Hyperparameter Tuning*](https://icml.cc/virtual/2020/poster/6589) at ICML 2020

In this we provide our PyTorch code: HYPerparameter Aware Optimizer Benchmarking Protocol (HYPAOBP) for our ICML 2020
paper. The pdf can be found [here](https://arxiv.org/abs/1910.11758).

The backbone code of this project comes mainly from the excellent
[**DeepOBS**](https://github.com/fsschneider/DeepOBS/tree/develop) package that provides several standard problems to
benchmark our optimizers. In the code, we use a slightly older version than the one on the **DeepOBS** site. It is quite
likely that multiple bug-fixes have been made on the main repository, that are not present here.

We provide a version of *DeepOBS* in the `deepobs` folder. We added two additional optimizers in the folder
`deepobs/pytorch/optimizers` for SGDM W<sup>C</sup>D and Adam W<sup>C</sup>D.

## Setup

We modify the **DeepOBS** and provide it as a submodule. To clone the submodule, run the following: 

```bash
git submodule init
git submodule update
```

The code was tested under Debian 9.0 and Python3.7, and we expect no surprises when used on other systems. The setup
should only require installing the packages specified in 'spec-file.txt' via.

```bash
conda create --name optimizerbenchmarking --file spec-file.txt
```

## Running experiments

The experiments involve substantial computation and each run can be triggered by a command like:

```python
python main_runner.py --problem <problem> --optim <optimizer> --algo random --random-seed 1337 --num-evals <budget> --log-path <log-path-root>/<optimizer-label>
```

Here, `<problem>` denotes a dataset/architecture combination, and must be specified as one of the following:

```text
'fmnist_2c2d'
'mnist_vae'
'quadratic_deep'
'fmnist_vae'
'cifar100_allcnnc'
'cifar10_3c3d'
'imdb_bilstm'
'svhn_wrn164'
'tolstoi_char_rnn'
```

The datasets download automatically into a local folder called `data_deepobs`, with the exception of `tolstoi`, to download which run the following command 

```sh
 sh deepobs/scripts/deepobs_prepare_data.sh -o=tolstoi
```

`<optim>` denotes an optimizer for training the dataset/architecture combination, and must be one of the following:

```text
'adam': Adam optimizer with learning rate, betas, and epsilon as hyperparameters.
'adamlr': Adam optimizer but only the learning rate is tunable, the rest is set to default values.
'adamlreps': Adam optimizer but only the learning rate and epsilon are tunable.
'adamwclr': AdamW optimizer, but with a constant weight decay parameter, and only the learning rate is tunable.
'adamwlr': AdamW optimizer, but only weight decay and learning rate are tunable.
'adagrad': Adagrad optimizer, tunable learning rate.
'sgd': Plain SGD, without momentum, weight decay, or learning rate schedules. Only the learning rate is tunable.
'sgdm': SGD with momentum, where both the learning rate and the momentum parameter are tunable.
'sgdmc': SGD with constant momentum, so that only the learning rate is tunable.
'sgdmcwc': SGD with constant momentum and weight decay. Only the learning rate is tunable.
'sgdmw': SGD with momentum and weight decay, all being tunable.
'sgdmcwclr': SGD with constant momentum and constant weight decay. The initial learning rate is tunable, but is decayed according to a "Poly" learning rate schedule.
'adamwclrdecay': AdamW with a constant weight decay. The initial learning rate is tunable, but is decayed according to a "Poly" learning rate schedule.
```

The `log-path` parameter specifies where to store the experiment's output, which is later used to generate statistics
and plots. In order to use our scripts out-of-the-box, you should specify the log path as
`<log-path-root>/<optimizer-label>`, where `<optimizer-label>` denotes a recognizable label of an optimizer, e.g., `SGD`
for `--optim sgd`. The `<budget>` parameter specifies the number of hyperparameter configurations drawn during random
search, e.g., in our paper, this number was set to 100.

We additionally provide an additional flag `--early-stopping` to enable early stopping. We use a logic that is very
similar to the one in [Keras](https://keras.io/api/callbacks/early_stopping/) with parameters ```patience = 2, min_delta
= 0.0, mode = 'min'```

## Analyzing experiments

Once all the experiments have been done, you can use our scripts to generate Tables and Figures from the paper.

### Summarizing across datasets

To generate the plot from Figure 3, which summarizes the performance of optimizers relative to the best performance
across datasets, you can use a command like the following:

```python
python compute_relative_performances.py -inpath <log-path-root> -optimizers <optimizer-label-list> -problems <problem-list> -budget <budget> -outfile <outfile> -num_shuffle <num-shuffle>
```

where `<log-path-root>` is specified as above, `<optimizer-label-list>` is a list of optimizer labels (e.g., `SGD` for
`sgd`) (must be same as used for main_runner.py), `<budget>` denotes the maximum budget to consider, `<outfile>` is the
path where to store the output figure. `<num-shuffle>` denotes the number of samples for the bootstrap method which
computes expected validation scores. The larger this parameter is chosen, the longer it takes to generate the figure.
For final results, a value > 1000 should be chosen. `<problem-list>` is a whitespace-separated list of problem
identifies (e.g., `fmnist_2c2d` from above). Make sure it only includes problems for which an experiment has been run
for all optimizers in `<optimizer-list>`.

### Boxplots

The boxplots from Figure 4 and 5 in the paper can be generated with the following command:

```python
python analyze_tunability.py -inpath <log-path-root> -optimizers <optimizer-label-list> -problem <problem> -outfile <outfile> -num_shuffle <num-shuffle>
```

The input parameters are similar to the summary plot script, but you have to specify a single problem instead of a list.

### Cumulative-Performance-Early (CPE)

To compute several scalar metrics, including the CPE scores from Table 6 in the paper, use the following command

```python
python analyze_tunability.py -inpath <log-path-root> -optimizers <optimizer-label-list> -problem <problem> -outfile <outfile> -num_shuffle <num-shuffle> -print_metrics
```

which is the same as for boxplots, but with the additional flag -print\_metrics.

### Stacked Probability Plot

For creating stacked probability plots like in Figures 1 and 6, execute the following commands in succession:

```python
python results_create_dataframe.py -inpath <log-path-root> -outfile <dataframe-path>
python results_process.py -inpath <dataframe-path> -budget <budget> -outfile_best_budget <outfile-path>
python results_analyze_winprob.py -inpath <outfile-path> -optimizers <optimizer-label-list> -budget <budget> -outfile <plot-path> -global_averaging
```

`<dataframe-path>` and `<outfile-path>` denote paths for storing intermediate files. `<plot-path>` stores the final
plot. By specifying the `global_averaging` flag, the probabilities across all datasets are combined, resulting in a plot
of the type from Figure 1 in the paper. If you leave it out, separate plots for each problem are generated. All problems
for which there are experiments found under `<log-path-root>` will be taken into consideration.

## Citation

If you find our work useful, or use our code, please cite us as:

```bibtex
@inproceedings{sivaprasad2020optimizer,
    title={Optimizer Benchmarking Needs to Account for Hyperparameter Tuning},
    author={Sivaprasad, Prabhu Teja and Mai, Florian and Vogels, Thijs and Jaggi, Martin and Fleuret, Francois},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    year = {2020}
}
```

## Credits

The code was written signficantly by [Prabhu Teja](mailto:prabhu.teja@idiap.ch) and [Florian Mai](mailto:fmai@idiap.ch),
with inputs from Thijs Vogels.
