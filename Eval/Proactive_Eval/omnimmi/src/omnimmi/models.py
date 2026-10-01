def build_offline(config):
    from vlmeval.vlm.qwen3_vl import Qwen3VLChat
    return Qwen3VLChat(model_path=config['model_path'],system_prompt=config['system_prompt'],
        min_pixels=config['min_pixels'],max_pixels=config['max_pixels'],total_pixels=config['total_pixels'],
        max_new_tokens=4096,temperature=0.,do_sample=False,top_p=1.,top_k=0,repetition_penalty=1.)
