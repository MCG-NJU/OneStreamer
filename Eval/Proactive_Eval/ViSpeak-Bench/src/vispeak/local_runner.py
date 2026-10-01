import time
from .model import Qwen3VLOnline
from .prompts import build_online_task_prompt

def infer(job,rows):
    model=Qwen3VLOnline(model_path=job['model_path'])
    for row in rows:
        target=row['conversations'][-1];reference=target.get('timespan',target.get('time'))
        end=float(reference[-1] if isinstance(reference,list) else reference)
        prompt=build_online_task_prompt(job['task'],row)
        session=model.create_session(prompt,float(row.get('video_start_time',0)))
        text=None;response_time=None;history=[]
        frames=model.iter_frames(row['media'],float(row.get('video_start_time',0)),end+2.)
        try:
            for frame,idx,observed in frames:
                start=time.monotonic();decision=model.step(session,frame,idx)
                history.append(dict(role='assistant',content=decision.raw_output,decision=decision.kind,time=observed,frame_interval=[idx,idx+1],cost=time.monotonic()-start))
                if decision.kind=='response': text=decision.response_text;response_time=observed;break
        finally: frames.close()
        yield dict(row,prediction=text,response_time=response_time,dialog_history=history,
                   task_prompt=prompt,evaluation_mode='online_proactive')

