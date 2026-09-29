"""Run unmodified OmniControl sampling with a fixed constraint set across seeds.

Only the construction of prompts/hints and the inference-only data loader are
wrapped. The model, weights, sampler, and all generated outputs are upstream.
This supplies matched candidates for an I1 feasibility test, not an I1 model.
"""

import argparse
import sys

import numpy as np

import sample.generate as official
from data_loaders.get_data import get_dataset_loader


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--r3_constraint_seed", type=int, required=True)
    opts, remaining = parser.parse_known_args()
    original_collate = official.collate_all

    def fixed_collate(*args):
        old_state = np.random.get_state()
        try:
            np.random.seed(opts.r3_constraint_seed)
            return original_collate(*args)
        finally:
            np.random.set_state(old_state)

    def inference_loader(**kwargs):
        kwargs["hml_mode"] = "text_only"
        return get_dataset_loader(**kwargs)

    official.collate_all = fixed_collate
    official.get_dataset_loader = inference_loader
    sys.argv = [sys.argv[0], *remaining]
    official.main()


if __name__ == "__main__":
    main()
