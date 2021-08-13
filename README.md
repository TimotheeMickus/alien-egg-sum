# Alien: the sum game

Repository for technical test: training a model to solve the sum game with the EGG library.

Here, we consider which of a regression loss or a classification loss is most appropriate to develop a communication protocol.

This repository contains:
 + the PDF of the report, as well as its source TeX code and visualization data
 + all necessary info to replicate the experiments: data, models, source code

The code is available in the `sum_game/` directory.
The `outputs/` directory contains productions from the two models, such as the messages they produced

## Installation

Install with:
```
python3 -m venv .venv && . .venv/bin/activate
pip3 install -r pip3.requirements.txt
```

## Usage

To find the best hyperparmeters for a regression model:
```
python3 sum_game/run.py \
    --do_hypertune \
    --game_type regression \
    --max_len 8 \
    --maxint 64 \
    --n_epochs 100 \
    --runs_per_conf 3 \
    --device cuda:0 \
    --gp_result_dump best_regression_model/gp_results_type-regression_maxlen-8.pkl \
    --tensorboard_dir .tb/regression_m8  \
    --save_dir best_regression_model
```

Likewise for a categorization model:
```
python3 sum_game/run.py
    --do_hypertune \
    --game_type categorization \
    --max_len 8 \
    --maxint 64 \
    --n_epochs 100 \
    --runs_per_conf 3 \
    --device cuda:0 \
    --gp_result_dump best_categorization_model/gp_results_type-categorization_maxlen-8.pkl \
    --tensorboard_dir .tb/categorization_m8  \
    --save_dir best_categorization_model
```

To perform a simple training, you can provide hyperparameters through the command line and remove the `--do_hypertune` flag.

To generate a dataset, use the `--do_generate_data` flag.
Note that it will overwrite any existing data stored in `.data/`.

To perform some basic analysis on a model, run:
```
python3 sum_game/analyze.py best_regression_model/ .data/*.txt
```
