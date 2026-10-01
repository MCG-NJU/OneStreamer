#!/usr/bin/env python3
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import sys
import time
sys.dont_write_bytecode = True
from common import ROOT,CODE_DIRS,PATHS,MODELS,resolve_path,read,read_rows,atomic,digest,code_fingerprint,model_fingerprint,setup_dataset_environment,annotation_files_fingerprint
from recipes import BENCHMARKS,jobs,records,protocol,partitions
from judge_client import settings as judge_settings
from score import needs_judge,score
from summarize import summarize
from common import validate_resources

def arguments():
    p=argparse.ArgumentParser(description='Sequential benchmark queue; named eval_queue to avoid stdlib queue shadowing.')
    p.add_argument('--model',choices=[*MODELS,'both'],default='OneStreamer-4B')
    p.add_argument('--bench',default='all',help='Comma-separated names or all')
    p.add_argument('--tasks',help='Comma-separated tasks within the chosen benchmark')
    p.add_argument('--setting',choices=['memory','recent16','both'],default='both')
    p.add_argument('--asr',choices=['with_asr','without_asr','both'],default='both')
    p.add_argument('--mode',choices=['infer','judge','all'],default='all')
    p.add_argument('--phase',choices=['prepare','memory','answer','score','all'],default='all',help='Optional OneStreamer stage')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--processes-per-gpu',type=int,choices=[1,2,3],default=1)
    p.add_argument('--require-eight',action='store_true')
    p.add_argument('--run-id')
    p.add_argument('--output-dir','--work-dir',dest='output_dir',default=PATHS['output_root'],help='Prediction root: ROOT/model/benchmark/run-id/profile/task')
    p.add_argument('--judge-output-dir',default=PATHS['judge_output_root'],help='Separate root for all scores and judge caches')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--limit',type=int,help='Per expanded job; omitted means full data')
    p.add_argument('--selection',choices=['first','shortest'],default='first')
    p.add_argument('--sample-ids',help='Comma-separated IDs, useful for targeted smoke tests')
    p.add_argument('--judge-model');p.add_argument('--judge-api-base');p.add_argument('--judge-workers',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit<=0:p.error('--limit must be positive')
    if args.phase=='score':args.mode='judge'
    if (args.resume or args.mode=='judge') and not args.run_id:p.error('--resume / --mode judge require --run-id')
    args.run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    if args.run_id in ('.','..') or Path(args.run_id).name!=args.run_id:p.error('--run-id must be a single directory name')
    benches=BENCHMARKS if args.bench=='all' else args.bench.split(',')
    if set(benches)-set(BENCHMARKS):p.error('Unknown benchmark')
    if args.phase in ('memory','answer') and set(benches)-{'ovobench','streamingbench'}:p.error('--phase is for OneStreamer benchmarks')
    if args.phase=='memory' and args.setting!='memory':p.error('--phase memory requires --setting memory')
    gpus=args.gpus.split(',')
    if len(gpus)!=len(set(gpus)) or any(not x.strip() for x in gpus):p.error('GPU list must be distinct and nonempty')
    if args.require_eight and args.mode!='judge' and len(gpus)!=8:p.error('The eight-GPU entry requires eight distinct GPUs')
    return args,benches,gpus

def run_workers(job,gpus,phase):
    directory=Path(job['directory']);processes=[];handles=[]
    try:
        for shard,ids in enumerate(job['shards']):
            if not ids:continue
            log=directory/'logs'/f'shard-{shard:03d}.log';log.parent.mkdir(parents=True,exist_ok=True)
            handle=log.open('a');handles.append(handle)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[shard%len(gpus)],PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
            command=[sys.executable,str(ROOT/'run_script/worker.py'),str(directory/'manifest.json'),str(shard),'--phase',phase]
            process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((process,shard))
        while any(p.poll() is None for p,_ in processes):
            failed=[(p,s) for p,s in processes if p.poll() not in (None,0)]
            if failed:raise RuntimeError(f'Worker {failed[0][1]} exited {failed[0][0].returncode}; see {directory}/logs')
            time.sleep(.5)
        failed=[(p,s) for p,s in processes if p.returncode!=0]
        if failed:raise RuntimeError(f'Worker {failed[0][1]} exited {failed[0][0].returncode}; see {directory}/logs')
    finally:
        for process,_ in processes:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        for process,_ in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for handle in handles:handle.close()

def merged(job,phase,write=True):
    directory=Path(job['directory']);prefix='memory' if phase=='memory' else 'predictions';rows=[]
    for shard,ids in enumerate(job['shards']):
        result=read_rows(directory/'shards'/f'{prefix}-{shard:03d}.jsonl')
        if len(result)!=len(ids) or {r['sample_id'] for r in result}!=set(ids):raise RuntimeError(f'Incomplete/duplicate shard {shard}')
        if any(r.get('status')!='ok' or r.get('fingerprint')!=job['fingerprint'] for r in result):raise RuntimeError('Invalid shard fingerprint/status')
        rows.extend(result)
    lookup={r['sample_id']:r for r in rows}
    if len(lookup)!=len(rows):raise RuntimeError('Cross-shard duplicate IDs')
    rows=[lookup[r['sample_id']] for r in job['samples']]
    if write:atomic(directory/(prefix+'.json'),rows)
    return rows

def load_saved_job(selection,directory):
    saved=read(directory/'manifest.json')
    fingerprint=saved.get('fingerprint')
    if not fingerprint or digest({k:v for k,v in saved.items() if k!='fingerprint'})!=fingerprint:
        raise RuntimeError(f'Invalid saved manifest fingerprint: {directory}')
    for key in ('model','bench','profile','task'):
        if saved.get(key)!=selection[key]:raise RuntimeError(f'Saved manifest {key} does not match the selected job')
    ids=[r['sample_id'] for r in saved['samples']]
    assigned=[sample_id for shard in saved['shards'] for sample_id in shard]
    if not ids or len(set(ids))!=len(ids) or len(assigned)!=len(ids) or set(assigned)!=set(ids):
        raise RuntimeError('Invalid saved sample/shard assignment')
    if saved['source_samples']<len(ids):raise RuntimeError('Invalid saved source sample count')
    return dict(saved,directory=str(directory))

def annotations_match(expected,actual):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and all(k in actual and annotations_match(v,actual[k]) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(actual,list) and len(expected)==len(actual) and all(annotations_match(a,b) for a,b in zip(expected,actual))
    return expected==actual

def saved_predictions(job):
    rows=merged(job,'all',write=False)
    if any(not annotations_match(sample,row) for sample,row in zip(job['samples'],rows)):
        raise RuntimeError('Prediction annotations differ from the saved manifest')
    return rows

def score_job(job,rows,judge_config,judge_report,code_hash):
    directory=Path(job['judge_directory']);group=directory.parents[1]
    atomic(directory/'manifest.json',dict(model=job['model'],benchmark=job['bench'],
        run_id=judge_report['run_id'],profile=job['profile'],task=job['task'],
        prediction_manifest=str(Path(job['directory'])/'manifest.json'),inference_fingerprint=job['fingerprint'],
        predictions_fingerprint=digest(rows),scoring_code_fingerprint=code_hash,
        judge=judge_config if needs_judge(job) else None))
    atomic(directory/'status.json',dict(judge_report,status='running'))
    refresh_summary(group)
    for name in ('metrics.json','judged.json','scored.json'):
        (directory/name).unlink(missing_ok=True)
    metrics=score(job,rows,judge_config)
    atomic(directory/'metrics.json',metrics)
    atomic(directory/'status.json',dict(judge_report,status='complete',metrics=metrics))
    refresh_summary(group)

def execute_judge_jobs(args,selections,output_root,judge_root):
    config=judge_settings(args);code_hash=code_fingerprint();preflight={}
    for selection in selections:
        directory=job_directory(output_root,selection,args.run_id)
        judge_directory=job_directory(judge_root,selection,args.run_id)
        report=dict(model=selection['model'],benchmark=selection['bench'],run_id=args.run_id,
            profile=selection['profile'],task=selection['task'],directory=str(judge_directory),
            prediction_directory=str(directory),samples=0,source_samples=0)
        try:
            job=load_saved_job(selection,directory);job['judge_directory']=str(judge_directory)
            report.update(samples=len(job['samples']),source_samples=job['source_samples'],shards=[len(s) for s in job['shards']])
            rows=saved_predictions(job)
            print(f"{job['model']} / {job['bench']} / {job['profile']} / {job['task']}: scoring {len(rows)} saved samples",flush=True)
            if args.dry_run:
                if needs_judge(job):report['judge']=score(job,rows,config,True)
                report['status']='prepared';preflight.setdefault(judge_directory.parents[1],[]).append(report)
            else:score_job(job,rows,config,report,code_hash)
        except BaseException as error:
            for name in ('metrics.json','judged.json','scored.json'):
                (judge_directory/name).unlink(missing_ok=True)
            atomic(judge_directory/'status.json',dict(report,status='failed',error_type=type(error).__name__,error=str(error)))
            refresh_summary(judge_directory.parents[1])
            raise
    for group,reports in preflight.items():atomic(group/'preflight.json',summarize(reports))
    print(f'Prediction root: {output_root}\nJudge root: {judge_root}\nRun ID: {args.run_id}',flush=True)

def media_fingerprint(rows):
    stats={}
    for row in rows:
        path=Path(row['media'])
        if str(path) in stats:continue
        if not path.exists():raise FileNotFoundError(path)
        if path.is_dir():
            stats[str(path)]=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(path.iterdir()) if p.is_file()]
        else:stats[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
    return digest(stats)

def output_roots(args):
    roots=resolve_path(args.output_dir),resolve_path(args.judge_output_dir)
    for root in roots:
        if root==ROOT or any(root==ROOT/name or ROOT/name in root.parents for name in CODE_DIRS):
            raise ValueError('Use a dedicated output folder; project/source directories cannot be output roots')
    if roots[0]==roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise ValueError('--output-dir and --judge-output-dir must be separate, non-nested directories')
    return roots

def job_directory(root,job,run_id):
    return root/job['model']/job['bench']/run_id/job['profile']/job['task']

def refresh_summary(group):
    reports=[read(path) for path in sorted(group.glob('*/*/status.json'))]
    atomic(group/'summary.json',summarize(reports))

def execute(args,benches,gpus):
    output_root,judge_root=output_roots(args)
    all_jobs=[];models=list(MODELS) if args.model=='both' else [args.model]
    for model in models:all_jobs.extend(jobs(benches,model,args.setting,args.asr,args.tasks.split(',') if args.tasks else None))
    if args.mode!='judge':validate_resources(all_jobs)
    if args.processes_per_gpu>1 and args.mode!='judge' and any(j['bench'] not in ('ovobench','streamingbench','omnimmi','ovo_timing') for j in all_jobs):
        raise ValueError('Use one process per GPU for this benchmark list')
    groups={job_directory(root,job,args.run_id).parents[1] for root in (output_root,judge_root) for job in all_jobs}
    with ExitStack() as stack:
        for group in sorted(groups):
            group.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context((group/'.queue.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='judge':execute_judge_jobs(args,all_jobs,output_root,judge_root)
        else:execute_jobs(args,all_jobs,gpus,output_root,judge_root)

def execute_jobs(args,all_jobs,gpus,output_root,judge_root):
    code_hash=code_fingerprint();judge_config=judge_settings(args)
    annotation_hashes={bench:annotation_files_fingerprint({bench}) for bench in {j['bench'] for j in all_jobs}}
    preflight={};model_hashes={model:model_fingerprint(MODELS[model]) for model in {j['model'] for j in all_jobs}}
    for job in all_jobs:
        directory=job_directory(output_root,job,args.run_id)
        judge_directory=job_directory(judge_root,job,args.run_id)
        group=directory.parents[1];judge_group=judge_directory.parents[1]
        cache=group/'dataset_cache'/digest([code_hash,annotation_hashes[job['bench']]])[:16];setup_dataset_environment(cache)
        source_rows=records(job);rows=source_rows
        if args.sample_ids:
            selected=set(args.sample_ids.split(','));rows=[r for r in rows if r['sample_id'] in selected]
            if {r['sample_id'] for r in rows}!=selected:raise ValueError('Unknown --sample-ids')
        if args.selection=='shortest':rows=sorted(rows,key=lambda r:(r['cost'],r['sample_id']))
        if args.limit:rows=rows[:args.limit]
        if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate samples')
        count=len(gpus)*args.processes_per_gpu
        if args.processes_per_gpu>1 and job['bench'] not in ('ovobench','streamingbench','omnimmi','ovo_timing'):raise ValueError('This benchmark supports one process per GPU')
        job.update(protocol=protocol(job),samples=rows,source_samples=len(source_rows),directory=str(directory),dataset_cache=str(cache),
            shards=partitions(rows,count,job['bench']=='streamingbench'),code_fingerprint=code_hash,
            model_fingerprint=model_hashes[job['model']],annotation_fingerprint=digest(source_rows),media_fingerprint=media_fingerprint(rows))
        if job['bench']=='omnimmi' and job['profile']=='with_asr':
            import hashlib
            job['asr_fingerprint']=digest({r['video']:hashlib.sha256((ROOT/'Proactive_Eval/omnimmi/data/asr_words/segments'/(r['video']+'.json')).read_bytes()).hexdigest() for r in rows})
        job['fingerprint']=digest(job)
        manifest=directory/'manifest.json'
        if manifest.exists():
            if read(manifest)['fingerprint']!=job['fingerprint']:raise RuntimeError(f'Run configuration changed: {directory}; use a new run ID')
            if not (args.resume or args.dry_run):raise RuntimeError('Existing run; use --resume or a new --run-id')
        else:atomic(manifest,job)
        job['judge_directory']=str(judge_directory)
        report=dict(model=job['model'],benchmark=job['bench'],run_id=args.run_id,profile=job['profile'],task=job['task'],samples=len(rows),source_samples=len(source_rows),shards=[len(s) for s in job['shards']],directory=str(directory))
        judge_report=dict(report,directory=str(judge_directory),prediction_directory=str(directory))
        print(f"{job['model']} / {job['bench']} / {job['profile']} / {job['task']}: {len(rows)}/{len(source_rows)} samples",flush=True)
        if args.dry_run or args.phase=='prepare':
            report['status']='prepared'
            preflight.setdefault(group,[]).append(report)
            continue
        active_directory=directory;active_report=report
        try:
            atomic(directory/'status.json',dict(report,status='running',fingerprint=job['fingerprint']))
            refresh_summary(group)
            run_workers(job,gpus,args.phase)
            rows=merged(job,args.phase)
            report['status']='memory_ready' if args.phase=='memory' else 'inferred'
            atomic(directory/'status.json',report)
            refresh_summary(group)
            if args.phase=='memory' or (needs_judge(job) and args.mode=='infer'):continue
            active_directory=judge_directory;active_report=judge_report
            score_job(job,rows,judge_config,judge_report,code_hash)
        except BaseException as error:
            atomic(active_directory/'status.json',dict(active_report,status='failed',error_type=type(error).__name__,error=str(error)))
            refresh_summary(active_directory.parents[1])
            raise
    for group,reports in preflight.items():atomic(group/'preflight.json',summarize(reports))
    print(f'Prediction root: {output_root}\nJudge root: {judge_root}\nRun ID: {args.run_id}',flush=True)

def main():
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    execute(*arguments())

if __name__=='__main__':main()
