"""Pinned example weights and browser assets; user uploads remain in the local cache."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CACHE = Path(os.environ.get('XGAUSSIAN_CACHE', ROOT / 'cache')).resolve()
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HUB_CACHE', str(CACHE / 'hub'))
MODEL_REPO = DATA_REPO = 'CaiYuanhao/X-Gaussian'
MODEL_REVISION = 'f46e4e1082f3731fbb9fe61c53dac2135e345337'
DATA_REVISION = '6f93c107a7c30d6264e3775b4eb64133d927def2'
RAW_CATALOG = json.loads((ROOT / 'catalog.json').read_text())
FIELD_PATHS = {name: path for item in RAW_CATALOG.values() for name, path in item['fields'].items()}


def download(path, *, dataset=False):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        DATA_REPO if dataset else MODEL_REPO, path,
        repo_type='dataset' if dataset else 'model',
        revision=DATA_REVISION if dataset else MODEL_REVISION,
        cache_dir=CACHE / 'hub',
    ))


def catalog():
    result = copy.deepcopy(RAW_CATALOG)
    for item in result.values():
        checkpoint = REPO / item['checkpoint']
        if not checkpoint.is_file():
            checkpoint = download(item['checkpoint'].removeprefix('pretrained/'))
        item['checkpoint'] = str(checkpoint)
        item['model'] = str(checkpoint.parents[2])
        item['fields'] = {name: str(CACHE / 'fields' / name) for name in item['fields']}
    return result


def field_file(name):
    if name not in FIELD_PATHS:
        raise KeyError(name)
    local = REPO / 'data' / FIELD_PATHS[name]
    return local if local.is_file() else download(FIELD_PATHS[name], dataset=True)
