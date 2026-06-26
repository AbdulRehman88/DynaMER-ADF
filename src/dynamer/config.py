from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _read_yaml(path: str | Path, required: bool = True) -> dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        if required:
            raise FileNotFoundError(f"YAML file not found: {fp}")
        return {}
    with fp.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {fp}")
    return data


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _flatten_vars(data: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)):
            values[str(key)] = str(value)
    return values


def _expand_string(value: str, variables: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            return variables[name]
        if name in os.environ:
            return os.environ[name]
        raise KeyError(
            f"Missing variable '{name}'. Define it in configs/local_paths.yaml "
            "or as an environment variable."
        )
    return _VAR_PATTERN.sub(replace, value)


def _expand_vars(obj: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(obj, str):
        return _expand_string(obj, variables)
    if isinstance(obj, list):
        return [_expand_vars(x, variables) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_vars(v, variables) for k, v in obj.items()}
    return obj


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


def _ensure_dirs(cfg: SimpleNamespace) -> None:
    for attr in [
        "output_dir",
        "cache_dir",
        "processed_dir",
        "log_dir",
        "table_dir",
        "figure_dir",
        "checkpoint_dir",
    ]:
        path = Path(getattr(cfg.paths, attr))
        path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path, local_paths_path: str | Path | None = None) -> SimpleNamespace:
    main_cfg = _read_yaml(config_path, required=True)

    local_paths: dict[str, Any] = {}
    if local_paths_path:
        local_paths = _read_yaml(local_paths_path, required=False)

    # PROJECT_ROOT defaults to the repository root if not supplied.
    repo_root = Path(config_path).resolve().parents[1]
    variables = {"PROJECT_ROOT": str(repo_root).replace("\\", "/")}
    variables.update(_flatten_vars(local_paths))

    expanded = _expand_vars(main_cfg, variables)
    expanded["_local_paths"] = local_paths

    cfg = _to_namespace(expanded)
    _ensure_dirs(cfg)
    return cfg
