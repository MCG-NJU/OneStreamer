import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import ROOT, resolve_path, read, read_rows, append, digest

def settings(args):
    config=read(ROOT/'run_script/configs/judge.json')
    config['api_base']=args.judge_api_base or os.environ.get('JUDGE_API_BASE') or config['api_base']
    config['model']=args.judge_model or config['model']
    if args.judge_workers: config['workers']=args.judge_workers
    return config

def evaluate_requests(requests, parse, max_tokens, directory, config, dry_run=False):
    extra={'enable_thinking':False} if 'dashscope.aliyuncs.com' in config['api_base'] else {'chat_template_kwargs':{'enable_thinking':False}}
    generation=dict(model=config['model'],temperature=0.,max_tokens=max_tokens,extra_body=extra)
    keyed=[(key,messages,digest(dict(messages=messages,api_base=config['api_base'],**generation))) for key,messages in requests]
    if len({key for key,_,_ in keyed})!=len(keyed): raise ValueError('Duplicate judge request ID')
    cache_path=Path(directory)/'judge_cache.jsonl'
    cached={r['fingerprint']:r for r in read_rows(cache_path) if r['status']=='ok'}
    pending=[r for r in keyed if r[2] not in cached]
    if dry_run: return dict(total=len(keyed),pending=len(pending),cached=len(keyed)-len(pending))
    if pending:
        from openai import OpenAI
        api_key=os.environ.get('JUDGE_API_KEY') or os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not api_key and os.environ.get('JUDGE_ENV_FILE'):
            from dotenv import dotenv_values
            env_path=resolve_path(os.environ['JUDGE_ENV_FILE'])
            if not env_path.is_file(): raise FileNotFoundError(f'JUDGE_ENV_FILE: {env_path}')
            private=dotenv_values(env_path)
            api_key=private.get('JUDGE_API_KEY') or private.get('DASHSCOPE_API_KEY') or private.get('OPENAI_API_KEY')
        if not api_key: raise RuntimeError('Judge requires JUDGE_API_KEY (or DASHSCOPE_API_KEY / OPENAI_API_KEY)')
        client=OpenAI(api_key=api_key,base_url=config['api_base'],timeout=config['timeout'],max_retries=0)
        def call(item):
            key,messages,fingerprint=item
            record=dict(request_id=key,fingerprint=fingerprint,status='failed')
            for attempt in range(config['retries']):
                raw=None
                try:
                    response=client.chat.completions.create(messages=messages,**generation)
                    raw=response.choices[0].message.content or ''
                    parsed=parse(raw)
                    if parsed is None: raise ValueError('Unparseable judge response')
                    return dict(record,status='ok',raw=raw,parsed=parsed,attempts=attempt+1)
                except Exception as error:
                    record.update(error_type=type(error).__name__,raw=raw,attempts=attempt+1)
                    if attempt+1<config['retries']: time.sleep(2**attempt)
            return record
        with ThreadPoolExecutor(max_workers=config['workers']) as pool:
            for future in as_completed([pool.submit(call,item) for item in pending]):
                result=future.result();append(cache_path,result)
                if result['status']=='ok': cached[result['fingerprint']]=result
        client.close()
    missing=[key for key,_,fp in keyed if fp not in cached]
    if missing: raise RuntimeError(f'{len(missing)} judge requests failed; inspect judge_cache.jsonl and resume')
    return {key:cached[fp] for key,_,fp in keyed}
