from common import ROOT, PATHS, MODELS, read, setup_dataset_environment

BENCHMARKS = ('ovobench', 'streamingbench', 'ovbench', 'odvbench',
              'proactivevideoqa', 'omnimmi', 'ovo_timing', 'vispeak')
VISPEAK_TASKS = ('Gesture_Understanding', 'Anomaly_Warning', 'Humor_Reaction',
                 'Visual_Interruption', 'Visual_Termination', 'Visual_Wake-Up')

def jobs(benchmarks, model, setting='both', asr='both', tasks=None):
    result = []
    for bench in benchmarks:
        configurations = [('paper', bench)]
        if bench == 'ovobench':
            configurations = [(s, bench) for s in ('memory', 'recent16') if setting in ('both', s)]
        elif bench == 'streamingbench': configurations = [('memory', bench)]
        elif bench == 'proactivevideoqa': configurations = [('paper', t) for t in ('EGO','TV','VAD','WEB')]
        elif bench == 'omnimmi':
            configurations = [(a, t) for a in ('with_asr','without_asr') if asr in ('both',a)
                              for t in ('ap','si','md','sg')] + [('cautious_two_protocol_v1','pa')]
        elif bench == 'vispeak': configurations = [('paper', t) for t in VISPEAK_TASKS]
        for profile, task in configurations:
            if tasks and task not in tasks: continue
            result.append(dict(bench=bench, model=model, model_path=MODELS[model],
                               profile=profile, task=task))
    if not result: raise ValueError('No jobs selected; check --bench and --tasks')
    if tasks and set(tasks) - {j['task'] for j in result}:
        raise ValueError(f'Unknown tasks: {set(tasks) - {j["task"] for j in result}}')
    return result

def dataset(job):
    bench = job['bench']
    if bench == 'ovbench':
        from vlmeval.dataset.ovbench_new import OVBenchBBox1000New
        return OVBenchBBox1000New(dataset='OVBench_BBox1000_New_2fps', fps=2,
            frames_limit=4096, ava_raw_fps=2, ava_raw_max_frames=64,
            ava_raw_window_seconds=120, ava_root=PATHS['ovbench_ava'],
            ava_anno_json=PATHS['ovbench_ava']+'/ovbench_ava_raw_bbox1000.json')
    if bench == 'odvbench':
        from vlmeval.dataset.odvbench_bbox1000 import ODVBenchBBox1000
        return ODVBenchBBox1000(dataset='ODVBench_BBox1000_4fps', fps=4, frames_limit=4096)
    if bench == 'proactivevideoqa':
        from vlmeval.dataset.proactive_videoqa import ProactiveVideoQA
        return ProactiveVideoQA(dataset='ProactiveVideoQA_'+job['task'], fps=2)
    if bench == 'ovo_timing':
        from vlmeval.dataset.ovo_timing import OVOTiming
        return OVOTiming(fps=4, max_num_frames=64)
    raise ValueError(bench)

def records(job):
    import pandas as pd
    bench = job['bench']
    if bench in ('ovobench','streamingbench'):
        filename = 'OVOBench_New_32frames.tsv' if bench == 'ovobench' else 'StreamingBench.tsv'
        data = pd.read_csv(PATHS[bench]+'/'+filename, sep='\t').fillna('')
        rows = []
        for item in data.to_dict('records'):
            idx = int(item['index'])
            row = {key:item[key] for key in ('task','subtask','video','start','end','question','answer','answer_text')}
            row.update(index=idx, sample_id=f'ovo-{idx:04d}' if bench=='ovobench' else f'streamingbench-realtime-{idx:04d}',
                options={k:str(item[k]).strip() for k in 'ABCDE' if k in item and str(item[k]).strip()},
                eval_mode=item.get('eval_mode','mcq_letter'))
            row['cost'] = float(row['end'])-float(row['start'])
            row['media'] = PATHS[bench]+'/'+row['video']
            rows.append(row)
        return rows
    if bench == 'omnimmi':
        from omnimmi.common import load_annotations, sample_id
        result = []
        for i, row in enumerate(load_annotations(PATHS[bench],job['task'])):
            row.update(index=i, sample_id=sample_id(job['task'],i), media=PATHS[bench]+'/videos/'+row['video'])
            row['cost'] = float(row.get('timestamp',0)) if job['task']=='si' else (float(row['answer'][1]) if job['task']=='pa' else len(row.get('qa',[]))+1)
            result.append(row)
        return result
    if bench == 'vispeak':
        rows=read(ROOT/'Proactive_Eval/ViSpeak-Bench/data/annotations'/f'{job["task"]}.json')
        for i,row in enumerate(rows):
            target=row['conversations'][-1];end=target.get('timespan',target.get('time'))
            end=float(end[-1] if isinstance(end,list) else end)
            row.update(index=i,sample_id=row['video'],media=PATHS[bench]+'/'+row['video'],cost=end-float(row.get('video_start_time',0))+2)
        return rows
    ds=dataset(job)
    rows=ds.data.fillna('').to_dict('records')
    for row in rows:
        row['sample_id']=str(row.get('index'))
        row['cost']=float(row.get('end',row.get('duration',0)) or 0)-float(row.get('start',0) or 0)
        if bench=='ovo_timing':row['cost']=float(row['end_time'])-(float(row['ask_time']) if row['task']=='CRR' else 0.)
        if bench=='proactivevideoqa':
            import json
            ends=[a['reply_timespan'][1] for a in json.loads(row['answer_json'])]
            row['cost']=min(row['cost'],max(ends)) if ends else row['cost']
        if bench in ('proactivevideoqa','odvbench'): row['media']=str(ds.data_root)+'/'+row['video']
        else:
            original=ds.data.loc[ds.data['index']==row['index']].iloc[0]
            if bench=='ovbench' and ds._is_ava_raw_line(original): row['media']=ds._resolve_ava_raw_path(original)
            else: row['media']=ds._resolve_media_path(original)[0]
    return rows

def protocol(job):
    bench=job['bench']
    if bench in ('ovobench','streamingbench'):
        return read(ROOT/'Proactive_Eval/OneStreamer/configs'/f'{bench}.json') | dict(recent_frames=16,recent_fps=1.,recent_resize_policy='source_pixels')
    if bench=='omnimmi':
        if job['task']=='pa': return dict(target_fps=1,max_rounds=32,max_tokens=128,min_pixels=3136,max_pixels=100352,
            temperature=0.,top_p=1.,top_k=0,system_prompt=(ROOT/'Proactive_Eval/omnimmi/pa_prompt.txt').read_text().rstrip('\n'))
        from omnimmi.asr import S0,S1
        return dict(protocol='recent_window',suite='omnimmi',condition='ABS_S1_P_before_local' if job['profile']=='with_asr' else 'ABS_S0_V',
            with_asr=job['profile']=='with_asr',system_prompt=S1 if job['profile']=='with_asr' else S0,
            min_pixels=None,max_pixels=None,total_pixels=None,max_new_tokens=4096,
            ap_lead_seconds=0.,recent_frames=32,target_fps=4.,budget_tokenizer=MODELS['qwen3vl-instruct'])
    if bench=='ovo_timing':
        from vlmeval.dataset.ovo_timing import _streaming_system_prompt
        return dict(fps=4,max_num_frames=64,max_new_tokens=128,temperature=.7,top_p=.8,top_k=20,
                    seed=42,repetition_penalty=1.,standby_high_res_frames=2,system_prompt=_streaming_system_prompt())
    if bench in ('proactivevideoqa','vispeak'):
        from shared.inference_fast import SYSTEM
        return dict(target_fps=2 if bench=='proactivevideoqa' else 1,max_rounds=64 if bench=='proactivevideoqa' else 32,
                    max_tokens=128,temperature=0.,top_p=1.,top_k=0,min_pixels=3136,max_pixels=100352,system_prompt=SYSTEM)
    return dict(fps=2 if bench=='ovbench' else 4,frames_limit=4096,min_pixels=784,max_pixels=200704,
                total_pixels=67108864,max_new_tokens=4096,temperature=0.,do_sample=False,
                top_p=1.,top_k=0,repetition_penalty=1.,seed=42,
                **(dict(ava_fps=2,ava_max_frames=64,ava_window_seconds=120,ava_min_start=900) if bench=='ovbench' else {}))

def partitions(rows,count,group_stream=False):
    groups={}
    for row in rows:
        key=(row['video'],row['start']) if group_stream else row['sample_id']
        groups.setdefault(key,[]).append(row)
    loads=[0.]*count; output=[[] for _ in range(count)]
    for group in sorted(groups.values(),key=lambda g:(-max(float(r['cost']) for r in g),g[0]['sample_id'])):
        idx=min(range(count),key=lambda i:(loads[i],i))
        output[idx].extend(r['sample_id'] for r in group)
        loads[idx]+=max(float(r['cost']) for r in group)
    return output
