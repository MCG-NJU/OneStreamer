import argparse
import importlib.util
from pathlib import Path
from common import read,read_rows,append,setup_dataset_environment,ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest');parser.add_argument('shard',type=int);parser.add_argument('--phase',default='all')
    args=parser.parse_args();job=read(args.manifest);job['phase']=args.phase
    setup_dataset_environment(job['dataset_cache'])
    directory=Path(job['directory']);prefix='memory' if args.phase=='memory' else 'predictions'
    path=directory/'shards'/f'{prefix}-{args.shard:03d}.jsonl'
    if path.exists():
        raw=path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            backup=path.with_suffix('.interrupted');backup.write_bytes(raw)
            path.write_bytes(raw[:raw.rfind(b'\n')+1])
    done=read_rows(path);seen={r['sample_id'] for r in done}
    if len(seen)!=len(done):raise ValueError('Duplicate shard records')
    assigned=set(job['shards'][args.shard])
    if not seen<=assigned or any(r['fingerprint']!=job['fingerprint'] for r in done):raise ValueError('Foreign shard cache')
    rows=[r for r in job['samples'] if r['sample_id'] in assigned-seen]
    if not rows:return
    import torch
    torch.cuda.init()
    if job['bench'] in ('ovobench','streamingbench'):from onestreamer.local_runner import infer
    elif job['bench']=='omnimmi':from omnimmi.local_runner import infer
    elif job['bench']=='vispeak':from vispeak.local_runner import infer
    else:
        spec=importlib.util.spec_from_file_location('local_vlmeval_runner',ROOT/'VLMEvalKit/runner.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);infer=module.infer
    expected={r['sample_id'] for r in rows}
    for row in infer(job,rows):
        if row['sample_id'] not in expected:raise ValueError('Unexpected or duplicate generated sample')
        append(path,dict(row,status='ok',fingerprint=job['fingerprint']))
        expected.remove(row['sample_id'])
        print(f"finished {job['model']} {job['bench']} {job['task']} {row['sample_id']}",flush=True)
    if expected:raise RuntimeError(f'Worker omitted {len(expected)} samples')

if __name__=='__main__':main()
