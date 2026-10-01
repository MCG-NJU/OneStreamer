def infer(job, rows):
    from recipes import dataset
    bench=job['bench']; p=job['protocol']
    ds=dataset(job)
    if bench=='proactivevideoqa':
        from vlmeval.vlm.qwen3_vl_online_stream import Qwen3VLOnlineStream
        model=Qwen3VLOnlineStream(model_path=job['model_path'],**p)
    elif bench=='ovo_timing':
        from vlmeval.vlm.qwen3_ovotiming import Qwen3OVOTiming
        model=Qwen3OVOTiming(model_path=job['model_path'],do_sample=True,**p)
    else:
        from vlmeval.vlm.qwen3_vl import Qwen3VLChat
        model=Qwen3VLChat(model_path=job['model_path'],**p)
    for item in rows:
        row=ds.data.loc[ds.data['index']==item['index']].iloc[0]
        message=ds.build_prompt(row,video_llm=True)
        prediction=model.generate(message,dataset=ds.dataset_name)
        if bench=='ovo_timing':
            from vlmeval.dataset.ovo_timing import validate_timing_prediction
            validate_timing_prediction(prediction,row,ds.expected_inference_config(row))
        yield dict(item,prediction=prediction)
