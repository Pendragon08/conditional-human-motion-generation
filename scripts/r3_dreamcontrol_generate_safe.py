"""Run the official DreamControl sampler without touching supplied references.

The official CLI loads HumanML3D in training mode even when all prompts are
provided. For this inference-only run, text_only supplies the same normalization
statistics without requiring training-set motion files or GloVe vectors.
"""

import argparse
import sys

import sample.generate as official
from data_loaders.get_data import get_dataset_loader


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--r3_limit", type=int, default=2)
    opts, remaining = parser.parse_known_args()
    if opts.r3_limit < 1 or opts.r3_limit > 100:
        raise ValueError("r3_limit must be between 1 and 100")

    original_collate = official.collate_all

    def limited_collate(*args):
        texts, hints = original_collate(*args)
        return texts[: opts.r3_limit], hints[: opts.r3_limit]

    def inference_loader(**kwargs):
        kwargs["hml_mode"] = "text_only"
        return get_dataset_loader(**kwargs)

    official.collate_all = limited_collate
    official.get_dataset_loader = inference_loader
    sys.argv = [sys.argv[0], *remaining]
    official.main()


if __name__ == "__main__":
    main()
