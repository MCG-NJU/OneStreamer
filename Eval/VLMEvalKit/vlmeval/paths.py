import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _load(name):
    values = json.loads((ROOT / 'run_script/configs' / name).read_text())
    return {key: str(resolve_path(value)) for key, value in values.items()}


PATHS = _load('paths.json')
MODELS = _load('models.json')


def dataset_root(key, environment):
    return str(resolve_path(os.environ.get(environment, '').strip() or PATHS[key]))
