"""Cache the public DistilBERT dependency used by CLoSD's diffusion planner."""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    repo = "distilbert/distilbert-base-uncased"
    path = Path(snapshot_download(
        repo_id=repo,
        allow_patterns=[
            "config.json", "pytorch_model.bin", "tokenizer.json",
            "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
        ],
        max_workers=2,
    ))
    required = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError("Missing CLoSD text-encoder files: " + ", ".join(missing))
    output = Path(os.environ["CLOSD_TEXT_ENCODER_MANIFEST"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "repo": repo,
        "revision": path.name,
        "path": str(path),
        "files": {name: (path / name).stat().st_size for name in required},
    }, indent=2) + "\n")
    print("CLOSD_TEXT_ENCODER_READY=" + str(output), flush=True)


if __name__ == "__main__":
    main()
