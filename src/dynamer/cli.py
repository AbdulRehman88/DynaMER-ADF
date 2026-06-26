from __future__ import annotations

import argparse
from pathlib import Path

from src.dynamer.config import load_config
from src.dynamer.utils.logging import get_logger
from src.dynamer.utils.seed import set_global_seed
from src.dynamer.data.inventory import run_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DynaMER command-line entry point")
    parser.add_argument("--config", type=str, required=True, help="Path to main YAML config.")
    parser.add_argument(
        "--local-paths",
        type=str,
        default="configs/local_paths.yaml",
        help="Optional YAML file containing local machine paths.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="inventory",
        choices=["inventory", "prepare", "train", "evaluate"],
        help="Pipeline stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.local_paths)
    logger = get_logger("DynaMER", log_dir=cfg.paths.log_dir)

    set_global_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
    logger.info("Loaded project: %s", cfg.project.name)
    logger.info("Stage: %s", args.stage)

    if args.stage == "inventory":
        run_inventory(cfg, logger)
    else:
        raise NotImplementedError(
            f"Stage '{args.stage}' is scaffolded but not implemented yet. "
            "We will implement it step-by-step after inventory is verified."
        )


if __name__ == "__main__":
    main()
