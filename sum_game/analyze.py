import argparse
import json
import pathlib

import egg.core as core
import torch

from archs import load_game
from data import SumGameStructuredDataset, SumGameOneHotDataset, to_dataloader

def all_topsim(game, dataset):
    with torch.no_grad():
        all_pair_embs, all_targets, all_pairs = next(iter(to_dataloader(dataset, batch_size=len(dataset))))
        messages = game.sender(all_pair_embs).argmax(dim=-1)
        return [
            core.language_analysis.TopographicSimilarity.compute_topsim(meanings, messages, meaning_distance_fn="euclidean")
            for meanings in (all_pair_embs, all_targets.unsqueeze(-1), all_pairs)
        ]

def stats(game, dataset):
    all_paired_inputs, all_targets, _ = next(iter(to_dataloader(dataset, batch_size=len(dataset))))
    loss, interaction = game(all_paired_inputs, all_targets)
    return loss, interaction.aux["acc"].mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--two_N", type=int, default=126)
    parser.add_argument("game_save_dir", type=pathlib.Path)
    parser.add_argument("dataset_paths", type=pathlib.Path, nargs="+")
    args = parser.parse_args()

    with open(args.game_save_dir / "config.json", "r") as istr:
        is_structured = json.load(istr)["ipt_format"] == "structured"

    game = load_game(args.game_save_dir / "config.json", args.game_save_dir / "model.pt")
    game.eval()
    for dataset_path in args.dataset_paths:
        if is_structured:
            dataset = SumGameStructuredDataset(dataset_path, args.two_N, keep_pairs=True)
        else:
            dataset = SumGameOneHotDataset(dataset_path, N=args.two_N // 2, keep_pairs=True)
        print(f"{args.game_save_dir} on {dataset_path.name}:")
        avg_loss, avg_acc = stats(game, dataset)
        print(f"\tL={avg_loss.item()}")
        print(f"\tacc={avg_acc.item()}")
        topsim_emb, topsim_sum, topsim_pair = all_topsim(game, dataset)
        print(f"\trho emb={topsim_emb}")
        print(f"\trho sum={topsim_sum}")
        print(f"\trho pair={topsim_pair}")
