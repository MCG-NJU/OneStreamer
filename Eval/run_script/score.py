from pathlib import Path
import re
from common import atomic
from judge_client import evaluate_requests

def needs_judge(job):
    return job['bench'] in ('proactivevideoqa','vispeak') or (job['bench']=='omnimmi' and job['task']!='pa')

def score(job, rows, judge_config, dry_run=False):
    bench=job['bench'];directory=Path(job['judge_directory'])
    if not dry_run:directory.mkdir(parents=True,exist_ok=True)
    if bench in ('ovobench','streamingbench'):
        from onestreamer.metrics import aggregate
        return aggregate(rows)
    if bench in ('ovbench','odvbench','ovo_timing'):
        import pandas as pd
        if bench=='odvbench':
            from vlmeval.utils.matching_util import can_infer
            hits=[];scored=[]
            for row in rows:
                choices={k:str(row[k]) for k in 'ABCDE' if row.get(k)}
                option=can_infer(str(row['prediction']),choices) or 'Z'
                hit=int(option==str(row['answer']));hits.append(hit);scored.append(dict(row,parsed_option=option,hit=hit))
            atomic(directory/'scored.json',scored)
            return dict(samples=len(hits),accuracy=sum(hits)/len(hits),aggregation='sample_micro_accuracy')
        if bench=='ovbench':
            from vlmeval.dataset.ovbench import OVBench
            ds=OVBench.for_scoring(job['samples'])
        else:
            from vlmeval.dataset.ovo_timing import OVOTiming
            ds=OVOTiming.for_scoring(job['samples'],job['protocol'])
        frame=ds.data.loc[ds.data['index'].isin([r['index'] for r in rows])].copy()
        pred={r['index']:r['prediction'] for r in rows}
        frame['prediction']=[pred[i] for i in frame['index']]
        ds.data = frame.drop(columns=['prediction']).copy()
        eval_path=directory/'predictions.tsv';frame.to_csv(eval_path,sep='\t',index=False)
        metrics=ds.evaluate(str(eval_path))
        if bench=='ovbench':
            per_task=metrics['sub_answer_type_accuracy']
            metrics['sample_micro_accuracy']=metrics.pop('overall_accuracy')
            metrics['paper_mean_16_subtasks']=sum(per_task.values())/16 if len(per_task)==16 else None
            metrics['covered_subtasks']=len(per_task)
        return metrics.to_dict('records') if hasattr(metrics,'to_dict') else metrics
    if bench=='omnimmi':
        from omnimmi.scoring import pa_metrics, _flatten_predictions, judge_messages, parse_judge_response, aggregate_main_scores
        if job['task']=='pa': return pa_metrics(rows)
        items=_flatten_predictions(job['task'],rows)
        requests=[(r['judge_id'],judge_messages(r['question'],str(r['answer']),r['prediction'])) for r in items]
        results=evaluate_requests(requests,parse_judge_response,128,directory,judge_config,dry_run)
        if dry_run:return results
        judged=[dict(r,judge=results[r['judge_id']]['parsed'],judge_status='ok') for r in items]
        atomic(directory/'judged.json',judged)
        return aggregate_main_scores(job['task'],judged)
    if bench=='proactivevideoqa':
        from vlmeval.dataset.pvqa_scoring import build_inputs,compute_scores
        requests,pred,gold=build_inputs(rows)
        def parse(raw):
            match=re.match(r'[123]',raw.strip())
            if not match: raise ValueError('Expected 1/2/3')
            return match.group()
        results=evaluate_requests(requests,parse,32,directory,judge_config,dry_run)
        if dry_run:return results
        scored,metrics=compute_scores(gold_dict=gold,pred_dict=pred,results={k:dict(content=v['parsed']) for k,v in results.items()})
        atomic(directory/'judged.json',scored)
        return dict(metrics,fallback_count=0,judge_requests=len(requests))
    if bench=='vispeak':
        from vispeak.scoring import response_is_in_time,build_prompt,parse_json_response,score_payload,task_summary
        kind=dict(Gesture_Understanding='gu',Anomaly_Warning='aw',Humor_Reaction='hr',Visual_Interruption='vi',Visual_Termination='vt',**{'Visual_Wake-Up':'vw'})[job['task']]
        requests=[];records={}
        for row in rows:
            target=row['conversations'][-1];ref=target.get('timespan',target.get('time'))
            correct=response_is_in_time(row['response_time'],ref)
            records[row['sample_id']]=dict(time_correct=correct,judge_status='pending' if correct else 'not_time_correct',score=0.,response_time=row['response_time'],reference_time=ref)
            if correct:
                prompt=build_prompt(kind,row['prediction'] or 'null',row)
                requests.append((row['sample_id'],[dict(role='user',content=prompt)]))
        def parse(raw):return score_payload(kind,parse_json_response(raw))
        results=evaluate_requests(requests,parse,256,directory,judge_config,dry_run)
        if dry_run:return results
        for key,result in results.items():records[key].update(judge_status='ok',score=result['parsed'])
        atomic(directory/'judged.json',records)
        return task_summary(records,len(rows))
    raise ValueError(bench)
