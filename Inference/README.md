# Proactive inference

A minimal example of **OneStreamer-4B** watching a video incrementally and deciding when to respond. Use the environment installed in [Eval](../Eval/README.md), and keep `Inference/` next to `Eval/` so this example can reuse the streaming SDK.

## Run

Download the OneStreamer-4B checkpoint into `../path_to/models/OneStreamer-4B`, or pass `--model-path`. From this directory:

```bash
conda activate onestreamer-eval
CUDA_VISIBLE_DEVICES=0 python inference.py
```

The included `assets/rec-1596.mp4` shows objects being presented to the camera. The default question asks the model to count these events as they happen. The script prints each `</Response>` immediately and writes all model updates to `outputs/responses.jsonl`.

To use your own video and question:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model-path ../path_to/models/OneStreamer-4B \
  --video ../path_to/videos/example.mp4 \
  --question "Tell me when the person picks up the cup." \
  --output outputs/my_video.jsonl
```

Relative paths are resolved from `Inference/`. Choose a new output filename for each run; existing results are not overwritten. Add `--show-all` to display `</Silence>` and `</Standby>` as well, or `--max-seconds 10` to process a short prefix.

## How it works

The script samples four frames per second and passes each one-second update to a `StreamingSession`. The session retains the model's own responses and a rolling window of up to 64 frames. One initial user request stays active throughout the video:

- `</Silence>`: keep watching.
- `</Standby>`: wait for more evidence.
- `</Response>`: deliver a proactive response.

The example uses the retained OVO-Timing prompt, 128 output tokens per update, seed 42, temperature 0.7, top-p 0.8 and top-k 20. It runs actual model inference; no responses are loaded from a saved trace. Processing speed depends on the GPU. Times in the output are sampled video intervals, rather than wall-clock latency; the last interval can extend slightly past the final video frame.

The example video comes from OVOBench's REC sample 1596. See [assets/README.md](assets/README.md) for provenance. Code is licensed under [Apache-2.0](LICENSE); video rights remain with its source.
