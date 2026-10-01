from pathlib import Path
import time
from .schema import ExperimentConfig,Sample
from .engine import OneStreamerEngine
from .cache import load_or_create_caption,load_or_create_shared_caption,_stream_key
from .memory import MemoryBank,MemoryEvent
from .video import read_recent_frames
from .prompts import MEMORY_QA_SYSTEM,MEMORY_DIRECT_QA_SYSTEM
from .ovobench import build_question_text,score_prediction

def infer(job, rows):
    from common import PATHS
    p=job['protocol'];phase=job.get('phase','all');memory=job['profile']=='memory'
    config=ExperimentConfig(path=Path(job['directory'])/'manifest.json',models={job['model']:job['model_path']},
        video_root=Path(PATHS[job['bench']]),samples_path=Path(job['directory'])/'samples.json',
        cache_root=Path(job['directory'])/'memory',caption=p['caption'],answer=p['answer'],variants=())
    engine=OneStreamerEngine(model_path=job['model_path'],dtype='bfloat16',attn_implementation='flash_attention_2',min_pixels=3136,max_pixels=100352)
    groups={}
    for row in job['samples']:
        sample=Sample.from_dict(row);groups.setdefault(_stream_key(sample),[]).append(sample)
    timeline_memory={}
    for row in rows:
        sample=Sample.from_dict(row);video=config.video_root/sample.video
        caption_data={'events':[],'counts':{}};cache_path=None
        if memory:
            args=dict(engine=engine,config=config,model_alias=job['model'],sample=sample,video_path=video,allow_create=phase!='answer')
            if job['bench']=='streamingbench':
                caption_data,_,cache_path=load_or_create_shared_caption(**args,stream_samples=groups[_stream_key(sample)],timeline_memory=timeline_memory)
            else: caption_data,_,cache_path=load_or_create_caption(**args)
        if phase=='memory':
            yield dict(row,caption_cache=str(cache_path),caption_counts=caption_data['counts']);continue
        bank=MemoryBank(MemoryEvent.from_dict(e) for e in caption_data['events'])
        mode='hierarchical' if memory else 'none'
        text,meta=bank.render_with_budget(mode=mode,question=sample.question,tokenizer=engine.processor.tokenizer,max_tokens=12000)
        frames=read_recent_frames(video,sample.start,sample.end,recent_frames=16,sample_fps=1.)
        start=time.monotonic()
        prediction=engine.answer(system=MEMORY_QA_SYSTEM if sample.eval_mode=='mcq_letter' else MEMORY_DIRECT_QA_SYSTEM,
            memory_text=text,recent_frames=frames,question_text=build_question_text(sample),max_tokens=16,recent_resize_policy='source_pixels')
        parsed,hit=score_prediction(sample,prediction)
        yield dict(row,model_alias=job['model'],variant='r16_fps1_source_'+mode,prediction=prediction,parsed_output=parsed,hit=hit,
            memory_text=text,memory_meta=meta,caption_cache=str(cache_path) if cache_path else None,caption_counts=caption_data['counts'],
            recent_timestamps=[t for t,_ in frames],recent_image_sizes=[list(im.size) for _,im in frames],
            runtime_seconds=time.monotonic()-start,**engine.last_input_metrics)
