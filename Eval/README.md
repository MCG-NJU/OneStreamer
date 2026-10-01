# Evaluation

Evaluation code for **OneStreamer-4B** and the **Qwen3-VL-4B-Instruct no-special-token** baseline. Both models use the same benchmark prompts and settings.

## Preparation

1. Create the evaluation environment and install dependencies from this directory. The required VLMEvalKit subset is included in this repository.

```bash
conda create -n onestreamer-eval python=3.10.9 -y
conda activate onestreamer-eval
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install packaging==25.0 psutil==7.2.2 ninja setuptools wheel
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-deps -r requirements-opencv.txt
```

The evaluated stack is Linux x86_64, CUDA 12.4, Transformers 4.57.6 and flash-attn 2.7.4.post1. Flash-attn requires a compatible wheel or CUDA development toolkit. The separate OpenCV installation preserves the evaluated OpenCV 4.13.0.92 / NumPy 1.26.4 combination. Known OpenCV and Decord metadata diagnostics are explained in [environment details](docs/environment.md).

2. Download the datasets for the eight benchmarks: **OVOBench, StreamingBench, OVBench, ODVBench, ProactiveVideoQA, OmniMMI, OVO-Timing, and ViSpeak**. OVO-Timing shares the OVOBench videos; OVBench also needs the New AVA data. Set their roots in [`run_script/configs/paths.json`](run_script/configs/paths.json). The expected files and directory structure are listed in [data layout](docs/data_layout.md). Selected annotations and the 721 OmniMMI ASR caches are already included.

3. Download the **OneStreamer-4B** and **Qwen3-VL-4B-Instruct no-special-token** checkpoints and set their paths in [`run_script/configs/models.json`](run_script/configs/models.json). By default, resources are placed under `../path_to/datasets/` and `../path_to/models/`, relative to this directory. ASR-enabled OmniMMI uses the Instruct tokenizer for either model.

## Usage

Run the eight-GPU evaluation queue:

```bash
# Optional CPU preflight: no model inference or judge API requests.
bash run_script/run_eval_8gpu.sh --model OneStreamer-4B --dry-run --run-id evaluation

# Inference; reuse the preflight run with --resume.
bash run_script/run_eval_8gpu.sh --model OneStreamer-4B --mode infer --run-id evaluation --resume
```

Change `--model` to `qwen3vl-instruct` or `both` to switch models. The queue runs jobs sequentially across eight GPUs, with one worker per GPU by default. Each model has 25 task configurations. OVOBench defaults to both memory and recent16 settings; OmniMMI defaults to the selected with-ASR and visual-only settings plus one PA run. See [evaluation settings](docs/evaluation_settings.md).

Single-benchmark launchers accept the same options:

| Benchmark | Launcher |
| --- | --- |
| OVOBench | `Proactive_Eval/OneStreamer/scripts/run_ovobench.sh` |
| StreamingBench | `Proactive_Eval/OneStreamer/scripts/run_streamingbench.sh` |
| OVBench | `VLMEvalKit/scripts/run_ovbench.sh` |
| ODVBench | `VLMEvalKit/scripts/run_odvbench.sh` |
| ProactiveVideoQA | `VLMEvalKit/scripts/run_proactivevideoqa.sh` |
| OmniMMI | `Proactive_Eval/omnimmi/run.sh` |
| OVO-Timing | `VLMEvalKit/scripts/run_ovo_timing.sh` |
| ViSpeak | `Proactive_Eval/ViSpeak-Bench/run.sh` |

```bash
bash Proactive_Eval/OneStreamer/scripts/run_ovobench.sh \
  --model OneStreamer-4B --setting recent16 --gpus 0 --limit 1 --mode infer --run-id example
```

All launchers use the active Python environment; `EVAL_PYTHON` can override it. Use `--resume` with the same inference settings, sample selection and shard count to continue an interrupted run. `--limit` applies per expanded task. OneStreamer memory tasks also support `--phase prepare|memory|answer|score|all`; memory-only preparation requires `--setting memory --phase memory`.

## Independent scoring

PVQA, OmniMMI QA and ViSpeak use a text judge. Supply `JUDGE_API_KEY`, `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`, or set `JUDGE_ENV_FILE=../path_to/secrets/judge.env`. The default service is Qwen3-235B-A22B on the public DashScope endpoint; settings are in [`judge.json`](run_script/configs/judge.json).

```bash
bash run_script/run_eval_8gpu.sh --model OneStreamer-4B --mode judge --run-id evaluation
```

Scoring needs the saved `manifest.json` and `shards/predictions-*.jsonl` for each selected task. It uses their saved annotations and protocols, and requires **no GPU, checkpoint, video or ASR cache**. To score on another machine, copy the prediction directory with its `<model>/<bench>/<run-id>/<setting>/<task>/` layout and pass it through `--output-dir`. The old absolute paths inside manifests are not reopened. Keep the model, benchmark, task and setting selectors matched to the saved run. Incomplete, duplicate or mismatched predictions are rejected.

`--mode judge --dry-run` validates saved inputs and reports pending API requests. Successful requests are cached; scoring failures remain failures. `--mode all` runs inference and scoring together. Local scores are computed during `--mode infer`; API-dependent scores require the judge stage.

## Outputs

```text
outputs/<model>/<bench>/<run-id>/<setting>/<task>/
judge_outputs/<model>/<bench>/<run-id>/<setting>/<task>/
```

Override these roots with `--output-dir` and `--judge-output-dir`. All relative resource and output paths are resolved from this evaluation directory. All metrics go under the judge root; each benchmark/run has a `summary.json`. Judge can use a new output root without rerunning inference and leaves prediction files unchanged.

Project code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for third-party components and data. Source provenance is recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).
