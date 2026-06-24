from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_CONFIG
from .models import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_CONFIG = PROJECT_ROOT / "config.local.yaml"
API_KEY_ENV_VARS = ("EISTARA_API_KEY", "EISTARA_LLM_API_KEY", "OPENAI_API_KEY")


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_mapping(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            data = parse_simple_yaml(text)
    return data if isinstance(data, dict) else {}


def parse_simple_yaml(text: str) -> dict:
    """Parse the small YAML subset used by Eistara config defaults.

    This fallback exists so Eistara can run before optional YAML
    dependencies are installed. It supports nested dictionaries via indentation
    plus scalar lists; full YAML remains delegated to PyYAML when available.
    """
    entries: list[tuple[int, str]] = []
    indentless_list_indent: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indentless_list_indent is not None:
            if indent == indentless_list_indent and stripped.startswith("- "):
                indent += 2
            else:
                indentless_list_indent = None
        entries.append((indent, stripped))
        if ":" in stripped and not stripped.startswith("- "):
            _key, value = stripped.split(":", 1)
            if value.strip() == "":
                indentless_list_indent = len(line) - len(line.lstrip(" "))

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for index, (indent, stripped) in enumerate(entries):
        if stripped.startswith("- "):
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if isinstance(parent, list):
                parent.append(parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            continue
        if value == "":
            child: dict[str, Any] | list[Any] = {}
            for next_indent, next_stripped in entries[index + 1 :]:
                if next_indent <= indent:
                    break
                child = [] if next_stripped.startswith("- ") else {}
                break
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"''", '""'}:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _nested_get(data: dict, path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _nested_set(data: dict, path: tuple[str, ...], value: Any) -> None:
    current = data
    for part in path[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[path[-1]] = value


def apply_secret_overrides(data: dict) -> dict:
    result = copy.deepcopy(data)
    for env_name in API_KEY_ENV_VARS:
        value = os.environ.get(env_name)
        if value and value.strip():
            _nested_set(result, ("api", "key"), value.strip())
            break
    return result


class ConfigLoader:
    def __init__(
        self,
        path: str | Path | None = None,
        defaults: dict | None = None,
        *,
        local_config: bool | str | Path = True,
    ):
        self.path = Path(path).expanduser().resolve() if path else None
        self.defaults = copy.deepcopy(defaults or DEFAULT_CONFIG)
        self.local_config = local_config
        self._cached_signature: tuple[Any, ...] | None = None
        self._cached_data: dict | None = None

    def load(self, force: bool = False) -> AppConfig:
        data = self.load_dict(force=force)
        return AppConfig.from_dict(data)

    def load_dict(self, force: bool = False) -> dict:
        signature = self._signature()
        if not force and self._cached_data is not None and signature == self._cached_signature:
            return copy.deepcopy(self._cached_data)
        local = load_mapping(self._local_config_path())
        merged = deep_merge(self.defaults, local)
        local_api_key = _nested_get(local, ("api", "key"))
        if self.path is not None:
            loaded = load_mapping(self.path)
            merged = deep_merge(merged, loaded)
            if local_api_key and not _nested_get(merged, ("api", "key")):
                _nested_set(merged, ("api", "key"), local_api_key)
        merged = apply_secret_overrides(merged)
        self._cached_signature = signature
        self._cached_data = merged
        return copy.deepcopy(merged)

    def get(self, key: str, default: Any = None) -> Any:
        data = self.load_dict()
        current: Any = data
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return copy.deepcopy(default)
        return copy.deepcopy(current)

    def _local_config_path(self) -> Path | None:
        if self.local_config is False:
            return None
        if self.local_config is True:
            return DEFAULT_LOCAL_CONFIG
        return Path(self.local_config).expanduser().resolve()

    def _signature(self) -> tuple[Any, ...]:
        return (
            self._path_signature(self.path),
            self._path_signature(self._local_config_path()),
            self._env_signature(),
        )

    def _path_signature(self, path: Path | None) -> tuple[int, int]:
        if path is None:
            return (0, 0)
        try:
            stat = path.stat()
        except OSError:
            return (0, 0)
        return (stat.st_mtime_ns, stat.st_size)

    def _env_signature(self) -> tuple[tuple[str, str], ...]:
        signature = []
        for name in API_KEY_ENV_VARS:
            value = os.environ.get(name) or ""
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""
            signature.append((name, digest))
        return tuple(signature)
