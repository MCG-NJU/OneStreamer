from collections import defaultdict

def summarize(reports):
    buckets=defaultdict(list)
    for r in reports:
        if r['status']=='complete':buckets[(r['model'],r['benchmark'])].append(r)
    output=[]
    for (model,bench),items in buckets.items():
        entry=dict(model=model,benchmark=bench,full_dataset=all(r['samples']==r['source_samples'] for r in items))
        if bench=='omnimmi':
            pa=next((r['metrics']['official_first_trigger_accuracy'] for r in items if r['task']=='pa'),None)
            for profile in ('with_asr','without_asr'):
                tasks={r['task']:r['metrics'].get('accuracy',r['metrics'].get('all_turns_sample_accuracy')) for r in items if r['profile']==profile}
                if pa is not None:tasks['pa']=pa
                output.append(dict(entry,profile=profile,task_accuracy=tasks,
                    mean_five_tasks=sum(tasks.values())/5 if set(tasks)=={'ap','si','md','sg','pa'} else None))
        elif bench=='proactivevideoqa':
            metrics={f'omega={w}':sum(r['metrics'][f'omega={w}'] for r in items)/4 if len(items)==4 else None for w in (0,.5,1)}
            output.append(dict(entry,mean_four_tasks=metrics,tasks_present=[r['task'] for r in items]))
        elif bench=='vispeak':
            fields=dict(time_all='time_accuracy',text_all='answer_average_score_time_correct',overall='answer_average_score_overall')
            output.append(dict(entry,**{k:sum(r['metrics'][v] for r in items)/6 if len(items)==6 else None for k,v in fields.items()},tasks_present=[r['task'] for r in items]))
        else:
            for r in items:output.append(dict(entry,profile=r['profile'],metrics=r['metrics']))
    return dict(jobs=reports,benchmarks=output,total_samples=sum(r['samples'] for r in reports))
