import argparse
import csv
import json
import pathlib

import egg.core as core
import torch

from archs import load_game
from data import SumGameStructuredDataset, SumGameOneHotDataset, to_dataloader


def all_topsim(game, dataset):
    with torch.no_grad():
        all_pair_embs, all_targets, all_pairs = next(
            iter(to_dataloader(dataset, batch_size=len(dataset)))
        )
        messages = game.sender(all_pair_embs).argmax(dim=-1)
        return [
            core.language_analysis.TopographicSimilarity.compute_topsim(
                meanings, messages, meaning_distance_fn="euclidean"
            )
            for meanings in (
                all_pair_embs,
                all_targets.unsqueeze(-1),
                all_pairs,
                all_pairs[:, 0].unsqueeze(-1),
                all_pairs[:, 1].unsqueeze(-1),
            )
        ]


def dump_messages(game, dataset, out_file, game_type):
    with torch.no_grad(), open(out_file, "w") as ostr:
        all_pair_embs, all_targets, all_pairs = next(
            iter(to_dataloader(dataset, batch_size=len(dataset)))
        )
        messages = game.sender(all_pair_embs).argmax(dim=-1)
        loss, interaction = game(all_pair_embs, all_targets)
        all_matches = interaction.aux["acc"].int()
        all_preds = interaction.receiver_output[:, -1, :]
        if game_type == "categorization":
            all_preds = all_preds.argmax(dim=-1)
        writer = csv.writer(ostr, delimiter="\t")
        for pair, s, pred, correct, msg in zip(
            all_pairs.int(), all_targets, all_preds, all_matches, messages
        ):
            _ = writer.writerow(
                [
                    pair[0].item(),
                    pair[1].item(),
                    s.item(),
                    pred.item(),
                    correct.item(),
                    *(w.item() for w in msg),
                ]
            )


def stats(game, dataset):
    all_paired_inputs, all_targets, _ = next(
        iter(to_dataloader(dataset, batch_size=len(dataset)))
    )
    loss, interaction = game(all_paired_inputs, all_targets)
    return loss, interaction.aux["acc"].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--two_N", type=int, default=126)
    parser.add_argument(
        "--msg_dump_dir", type=pathlib.Path, default=pathlib.Path("messages")
    )
    parser.add_argument("game_save_dir", type=pathlib.Path)
    parser.add_argument("dataset_paths", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    with open(args.game_save_dir / "config.json", "r") as istr:
        config_opts = json.load(istr)
        is_structured = config_opts["ipt_format"] == "structured"
        game_type = config_opts["game_type"]
    args.msg_dump_dir.mkdir(exist_ok=True, parents=True)
    game = load_game(
        args.game_save_dir / "config.json", args.game_save_dir / "model.pt"
    )
    game.eval()
    for dataset_path in args.dataset_paths:
        if is_structured:
            dataset = SumGameStructuredDataset(
                dataset_path, args.two_N, keep_pairs=True
            )
        else:
            dataset = SumGameOneHotDataset(
                dataset_path, N=args.two_N // 2, keep_pairs=True
            )
        print(f"{args.game_save_dir} on {dataset_path.name}:")
        avg_loss, avg_acc = stats(game, dataset)
        print(f"\tL={avg_loss.item()}")
        print(f"\tacc={avg_acc.item()}")
        topsim_emb, topsim_sum, topsim_pair, topsim_a, topsim_b = all_topsim(
            game, dataset
        )
        print(f"\trho emb={topsim_emb}")
        print(f"\trho sum={topsim_sum}")
        print(f"\trho pair={topsim_pair}")
        print(f"\trho a={topsim_a}")
        print(f"\trho b={topsim_b}")
        dump_messages(
            game,
            dataset,
            args.msg_dump_dir / dataset_path.with_suffix(".tsv").name,
            game_type,
        )
