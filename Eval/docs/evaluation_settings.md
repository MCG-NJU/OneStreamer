# Retained evaluation settings

`OneStreamer-4B` and `qwen3vl-instruct` use identical prompts, sampling, generation and scoring for each job. Instruct is the no-special-token checkpoint; protocol strings are supplied as text without registering or training new tokens. The executable settings are in `run_script/recipes.py` and the two OneStreamer JSON configurations.

## OVOBench and StreamingBench

OVOBench uses the OneStreamer evaluator with two settings:

- `memory`: hierarchical memory plus a recent visual window.
- `recent16`: no memory generation or caption cache; 16 recent frames sampled at 1 fps, retaining source pixels.

StreamingBench uses hierarchical memory and a shared causal timeline per video/start group. Questions only consume memory available at their own endpoint.

Memory generation uses 4 fps, a 128-frame context, one update per second, `silence_observe_summary_v50_training_user`, legacy dynamic resizing, greedy decoding and 128 generated tokens per update. EOF summary flushing remains conditional in the frozen implementation. The recent answer window is 16 frames at 1 fps, source resolution, with 16 answer tokens and a 12,000-token memory budget. StreamingBench's shared timeline keeps its configured 256-token summary limit. Both use the retained local answer scoring.

## OVBench and ODVBench

OVBench runs `OVBench_BBox1000_New_2fps`. Non-AVA data use 2 fps. New AVA uses raw video at 2 fps, at most 64 frames from a 120-second window, with the minimum start at 900 seconds for later timestamps. Score aggregation preserves the 16-task macro average.

ODVBench runs `ODVBench_BBox1000_4fps` and retains exact-match scoring. Both QA adapters use a 4096-frame ceiling, 784 minimum pixels, 200704 maximum pixels, 67108864 total pixels, greedy decoding, at most 4096 generated tokens, repetition penalty 1 and seed 42.

## ProactiveVideoQA

The EGO/TV/VAD/WEB subsets use the shared three-protocol streaming SDK: 2 fps, 64 rounds, 128 generated tokens per round, greedy decoding, and no future-frame prefetch. Pixel bounds are 3136–100352. The Qwen text judge uses the original 1/2/3 prompt and at most 32 output tokens. PAUC at ω=0/0.5/1 and the equal-weight four-subset aggregation are retained.

## OmniMMI

QA includes AP, SI, MD and SG. Retained conditions are exactly `ABS_S1_P_before_local` (with ASR) and `ABS_S0_V` (visual-only): recent 32 frames at 4 fps, absolute timestamps, no explicit min/max/total pixel overrides, greedy decoding, at most 4096 new tokens, and AP lead time 0. MD and SG feed each model's own previous predictions into the next turn.

ASR is offline transcription of the full audio followed by causal word selection: retain words whose `end` falls within the current local window according to the selected implementation, preserve their order, and put the P text block before the video. The lexical budget is 4096 tokens using the Instruct tokenizer for both model identities. Only the 721 selected caches are bundled; alternate ASR organization experiments are absent. The visual-only condition skips ASR files and budget-tokenizer loading.

PA uses **`cautious_two_protocol_v1`** and `Proactive_Eval/omnimmi/pa_prompt.txt`, 1 fps, 32 rounds, 128 generated tokens, pixel bounds 3136–100352, greedy decoding, and **no ASR**. It is scored locally by first-trigger accuracy, once per model. The same PA result is included alongside either QA condition.

QA judging preserves the yes/no plus 0–5 protocol with at most 128 tokens. AP/SI aggregate by sample; MD/SG count a sample as correct only when all its turns are correct.

## OVO-Timing

Timing retains its independent SDK and conservative three-protocol prompt. Use 4 fps, at most 64 frames, 128 generated tokens, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1, seed 42 and `standby_high_res_frames=2`. CRR/REC/SSR retain their original task logic and local metrics; timing is not routed through the shared proactive SDK.

## ViSpeak

Six tasks are retained: Gesture Understanding, Anomaly Warning, Humor Reaction, Visual Interruption, Visual Termination and Visual Wake-Up. They use the shared streaming SDK at 1 fps, 32 rounds, 128 generated tokens, pixel bounds 3136–100352 and greedy decoding. The temporal tolerance is ±2 seconds. Only temporally correct responses go to the content judge (0–5, at most 256 tokens); a temporal miss receives a local zero.

## Queue and scoring

The default matrix expands to 25 jobs and 26,325 sample configurations per model, or 50 jobs for both. All judge prompts are preserved. Text judging defaults to Qwen3-235B-A22B via the public DashScope OpenAI-compatible API, temperature 0 and thinking disabled. Successful requests are cached; failures remain failures. Partial-data metrics are marked as such and do not represent full benchmark results.
