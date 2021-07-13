# Global variables are defined here. We're also setting the global egg variables while we're at it.

import argparse
import pathlib

import egg.core as core

__maxint = 200
parser = argparse.ArgumentParser()
parser.add_argument("--generate_data", action="store_true")
parser.add_argument("--maxint", default=__maxint, type=int)
parser.add_argument("--n_features", default=__maxint * 2, type=int)
parser.add_argument("--n_hidden", default=3, type=int)
parser.add_argument("--embed_dim", default=7, type=int)
parser.add_argument("--cell", default="RNN", type=str)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--data_dir", default=pathlib.Path('.data'), type=pathlib.Path)
defaults = vars(core.init(parser))
print(defaults)
globals().update(defaults)
