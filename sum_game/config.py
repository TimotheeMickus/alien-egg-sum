import argparse
import pathlib
import pprint

import skopt
import torch
import egg.core as core


def get_search_space():
    search_space = [
        skopt.space.Integer(1, 10, "uniform", name="embed_pow"),
        skopt.space.Integer(
            1, 10, "uniform", name="hidden_pow"
        ),
        skopt.space.Integer(
            0, 10, "uniform", name="batch_pow"
        ),
        skopt.space.Integer(4, 64, "log-uniform", name="vocab_size"),
        skopt.space.Real(
            1e-8, 1.0, "log-uniform", name="lr"
        ),
        skopt.space.Real(
            0.0, 1.0, "uniform", name="entropy_coeff"
        ),
        skopt.space.Real(
            1e-8, 10, "log-uniform", name="temperature"
        ),
        skopt.space.Categorical(["rnn", "gru", "lstm"], name="cell"),
        skopt.space.Categorical(["reinforce", "gs"], name="mechanism"),
        skopt.space.Categorical([True, False], name="use_curriculum"),
        skopt.space.Integer(2, 7, name="n_updates"),
        skopt.space.Integer(8, 100, name="curriculum_length"),
    ]
    return search_space


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_generate_data", action="store_true")
    parser.add_argument("--do_hypertune", action="store_true")
    parser.add_argument(
        "--game_type",
        choices=("categorization", "regression"),
        default="categorization",
    )
    parser.add_argument("--mechanism", choices=("gs", "reinforce"),default="reinforce",)
    parser.add_argument("--maxint", default=512, type=int)
    # parser.add_argument("--n_features", default=__maxint * 2, type=int)
    parser.add_argument("--n_hidden", default=10, type=int)
    parser.add_argument("--embed_dim", default=10, type=int)
    parser.add_argument("--cell", default="rnn", type=str)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--data_dir", default=pathlib.Path(".data"), type=pathlib.Path)
    parser.add_argument("--device", default=torch.device("cuda:0"), type=torch.device)
    parser.add_argument(
        "--save_dir", default=pathlib.Path("best_model"), type=pathlib.Path
    )
    parser.add_argument("--runs_per_conf", default=3, type=int)
    parser.add_argument("--entropy_coeff", default=1e-1, type=float)
    parser.add_argument("--gp_result_dump", default="skopt_gp.pkl", type=pathlib.Path)
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--n_updates", default=7, type=int)
    parser.add_argument("--curriculum_length", default=20, type=int)
    args = core.init(parser)
    print(f"args:")
    pprint.pprint(vars(args))
    return args
