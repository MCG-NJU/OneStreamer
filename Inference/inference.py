#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def resource_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def arguments():
    parser = argparse.ArgumentParser(description='Run OneStreamer-4B on a video and print proactive responses as they are generated.')
    parser.add_argument('--model-path', default='../path_to/models/OneStreamer-4B')
    parser.add_argument('--video', default='assets/rec-1596.mp4')
    parser.add_argument('--question', default='How many times does the event: showing something to the camera happen?')
    parser.add_argument('--output', default='outputs/responses.jsonl')
    parser.add_argument('--show-all', action='store_true', help='Also print Silence and Standby updates')
    parser.add_argument('--max-seconds', type=float, help='Optional prefix of the video to process')
    args = parser.parse_args()
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error('--max-seconds must be finite and positive')
    if not args.question.strip():
        parser.error('--question must not be empty')
    return args


def video_windows(path, max_seconds=None, fps=4):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or frame_count <= 0:
            raise RuntimeError('The video must report a positive frame rate and frame count')
        duration = frame_count / source_fps
        end = min(duration, max_seconds) if max_seconds is not None else duration
        frames, times = [], []
        for step in range(math.ceil(end * fps)):
            timestamp = step / fps
            frame_index = min(int(timestamp * source_fps), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode frame at {timestamp:.3f}s')
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            times.append((timestamp, timestamp + 1 / fps))
            if len(frames) == fps:
                yield frames, times
                frames, times = [], []
        if frames:
            yield frames, times
    finally:
        cap.release()


def run(args):
    model_path, video_path, output_path = map(resource_path, (args.model_path, args.video, args.output))
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f'Provide a OneStreamer-4B checkpoint through --model-path: {model_path}')
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if output_path.exists():
        raise FileExistsError(f'Choose a new --output path: {output_path}')
    eval_kit = ROOT.parent / 'Eval/VLMEvalKit'
    if not eval_kit.is_dir():
        raise FileNotFoundError('Keep Inference/ and Eval/ next to each other')
    sys.path.insert(0, str(eval_kit))

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This example requires a CUDA GPU')
    torch.cuda.init()
    from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat
    from vlmeval.vlm.qwen3_ovotiming.inference_fast import StreamingSession
    from vlmeval.vlm.qwen3_ovotiming.utils import (
        _Qwen3VLInferenceFastEngine, processor_visual_geometry, smart_video_resize,
    )
    from vlmeval.dataset.ovo_timing import _streaming_system_prompt

    print('Loading OneStreamer-4B...', flush=True)
    model = Qwen3VLChat(
        model_path=str(model_path), seed=42, temperature=0.7, top_p=0.8,
        top_k=20, repetition_penalty=1.0, do_sample=True, max_new_tokens=128,
    )
    engine = _Qwen3VLInferenceFastEngine(model)
    _, _, resize_factor = processor_visual_geometry(model.processor)
    session = StreamingSession(
        engine, system=_streaming_system_prompt(), question=args.question,
        question_time=0, global_question=True, max_num_frames=64,
        max_tokens=128, temperature=0.7,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Question: {args.question}', flush=True)
    responses = 0
    try:
        with output_path.open('x', encoding='utf-8') as handle:
            for round_index, (frames, times) in enumerate(video_windows(video_path, args.max_seconds)):
                resized = []
                for frame in frames:
                    height, width = smart_video_resize(
                        num_frames=1, height=frame.height, width=frame.width,
                        factor=resize_factor, frame_min_pixels=resize_factor ** 2,
                        frame_max_pixels=224 * 224, force_resize=True,
                    )
                    resized.append(frame.resize((width, height)))
                answer = session.step_frames(resized, times, round_index) or ''
                is_response = '</Response>' in answer
                responses += int(is_response)
                record = dict(start=times[0][0], end=times[-1][1], raw=answer,
                              response=answer.split('</Response>', 1)[-1].strip() if is_response else None)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                if is_response or args.show_all:
                    print(f'[{times[-1][1]:.2f}s] {answer}', flush=True)
    finally:
        session.reset()
    print(f'Finished: {responses} proactive responses. Saved all updates to {output_path}', flush=True)


if __name__ == '__main__':
    run(arguments())
