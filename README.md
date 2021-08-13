# Alien: the sum game

Repository for technical test: training a model to solve the sum game with the EGG library.

Here, we consider which of a regression loss or a classification loss is most appropriate to develop a communication protocol.

This repository contains:
 + the PDF of the report, as well as its source TeX code and visualization data
 + all necessary info to replicate the experiments: data, models, source code

## Installation

Install with:
```
$ python3 -m venv .venv && . .venv/bin/activate
(.venv) $ pip3 install -r pip3.requirements.txt
```

## Usage

To train both models and find the best hyperparmeters:
```
(.venv) $ python3 sum_game/run.py --max_len 8 --maxint 64 --n_epochs 100 --runs_per_conf 3 --device cuda:0 --do_hypertune --game_type regression --gp_result_dump best_regression_model/gp_results_type-regression_maxlen-8.pkl --tensorboard_dir .tb/regression_m8  --save_dir best_regression_model
(.venv) $ python3 sum_game/run.py --max_len 8 --maxint 64 --n_epochs 100 --runs_per_conf 3 --device cuda:0 --do_hypertune --game_type categorization --gp_result_dump best_categorization_model/gp_results_type-categorization_maxlen-8.pkl --tensorboard_dir .tb/categorization_m8  --save_dir best_categorization_model
```

Note: the `--do_generate_data` flag will overwrite any existing data stored in `.data/`.

To perform some basic analysis on a model, run:
```
(.venv) $ python3 sum_game/analyze.py best_regression_model/ .data/*.txt
```
