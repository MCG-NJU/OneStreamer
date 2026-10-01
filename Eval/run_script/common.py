from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ('VLMEvalKit', 'Proactive_Eval', 'run_script')
for relative in ('VLMEvalKit', 'Proactive_Eval', 'Proactive_Eval/OneStreamer/src',
                 'Proactive_Eval/omnimmi/src', 'Proactive_Eval/ViSpeak-Bench/src'):
    sys.path.insert(0, str(ROOT / relative))
sys.dont_write_bytecode = True

def read(path):
    return json.loads(Path(path).read_text())

from vlmeval.paths import PATHS, MODELS, resolve_path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def read_rows(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []

def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

def code_fingerprint():
    paths = list(ROOT.iterdir())
    for name in CODE_DIRS:
        paths.extend((ROOT / name).rglob('*'))
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(paths) if p.is_file() and
                   (p.suffix in ('.py', '.sh', '.txt') or 'configs' in p.relative_to(ROOT).parts)})

def model_fingerprint(path):
    root = Path(path)
    if not (root / 'config.json').is_file():
        raise FileNotFoundError(root / 'config.json')
    weights = sorted(root.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError(f'No safetensors weights under {root}')
    index = root / 'model.safetensors.index.json'
    if index.exists():
        for filename in set(read(index)['weight_map'].values()):
            if not (root / filename).is_file(): raise FileNotFoundError(root / filename)
    return digest({'metadata': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(root.glob('*.json'))},
                   'weights': [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in weights]})

def validate_resources(jobs):
    keys = {('ovobench' if j['bench']=='ovo_timing' else j['bench']) for j in jobs}
    if 'ovbench' in keys: keys.add('ovbench_ava')
    models = {j['model'] for j in jobs}
    if any(j['bench']=='omnimmi' and j['profile']=='with_asr' for j in jobs):
        models.add('qwen3vl-instruct')
    for filename, values, selected in (('paths.json', PATHS, keys), ('models.json', MODELS, models)):
        for key in sorted(selected):
            if not Path(values[key]).is_dir():
                raise FileNotFoundError(f'run_script/configs/{filename}: {key} -> {values[key]} (missing directory)')

def annotation_files_fingerprint(benchmarks):
    paths = list((ROOT / 'VLMEvalKit/json_data').rglob('*.json'))
    if 'ovbench' in benchmarks:
        paths += [Path(PATHS['ovbench_ava']) / 'ovbench_ava_raw_bbox1000.json']
    if 'proactivevideoqa' in benchmarks:
        paths += [Path(PATHS['proactivevideoqa']) / task / 'anno.json' for task in ('EGO','TV','VAD','WEB')]
    return digest({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

def setup_dataset_environment(cache):
    for key in list(os.environ):
        if key.startswith(('OVBENCH_', 'ODVBENCH_', 'OVOBENCH_', 'OVOTIMING_', 'PROACTIVE_')):
            os.environ.pop(key)
    os.environ.update(LMUData=str(cache), PRED_FORMAT='tsv',
        OVBENCH_ROOT=PATHS['ovbench'], ODVBENCH_ROOT=PATHS['odvbench'],
        OVOBENCH_ROOT=PATHS['ovobench'], PROACTIVE_VIDEOQA_ROOT=PATHS['proactivevideoqa'],
        OVBENCH_MAX_FRAMES_NUM='4096', OVOTIMING_MAX_NUM_FRAMES='64', OVOTIMING_STRICT='1')
