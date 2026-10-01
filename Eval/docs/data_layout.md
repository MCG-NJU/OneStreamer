# Data and checkpoint layout

The external roots below are relative to the project root. Their default parent is `../path_to/datasets/`. Edit `run_script/configs/paths.json` to point at equivalent data on your machine. Media names and directory structure within each dataset must match the annotations; changing the outer root does not rename internal paths.

| Configuration key | Default directory | Required contents |
| --- | --- | --- |
| `ovobench` | `OVOBench/` | `OVOBench_New_32frames.tsv` and every relative `video` it references; ID-based Timing videos such as `<sample_id>.mp4` |
| `streamingbench` | `StreamingBench/` | `StreamingBench.tsv` and its referenced media, typically `videos/` |
| `ovbench` | `OVBench/` | Media referenced by the packaged BBox1000 annotation, including ArgoVerse, BDD, COIN, Charades, HACS, LaSOT, YFCC100M and hirest directories |
| `ovbench_ava` | `OVBench_new/` | `ovbench_ava_raw_bbox1000.json` and `AVA_RAW/` videos; use the BBox1000 annotation |
| `odvbench` | `ODVBench/` | Media tree corresponding to OMQA-DS-ALL: `TOI_Recognition/`, `TR_Analysis/`, `TS_Retrieval/` |
| `proactivevideoqa` | `ProactiveVideoQA/` | `EGO/`, `TV/`, `VAD/`, `WEB/`, each containing `anno.json` and `videos/` |
| `omnimmi` | `OmniMMI/` | Five selected task annotation JSONs and `videos/` (details below) |
| `vispeak` | `ViSpeak-Bench/` | Media directories directly beneath this root: `Gesture_Understanding/`, `Anomaly_Warning/`, `Humor_Reaction/`, `Visual_Interrupt/`, `Visual_Wake-Up_and_Termination/`, and all other relative paths referenced by the packaged annotations |

OVOBench QA and OVO-Timing share one root. Timing uses the packaged `VLMEvalKit/json_data/ovobench/ovobench_timing.json`; its resolver supports ID-based videos in the root, `chunked_videos/`, task directories, or the retained `video/` layouts. Keep the layout matching your annotation source. ViSpeak's configured root must directly contain the media task directories; do not add an extra nested `ViSpeak-Bench` level.

## OmniMMI annotations and ASR

Place these files directly under `OmniMMI/` or under `OmniMMI/annotations/`:

| Task | Filename |
| --- | --- |
| AP | `action_prediction.json` |
| SI | `speaker_identification.json` |
| MD | `multiturn_dependency_reasoning.json` |
| SG | `dynamic_state_grounding.json` |
| PA | `proactive_alerting.json` |

All media are resolved as `OmniMMI/videos/<video>`. Only these five tasks are evaluated. The four QA tasks use the selected word-level caches in `Proactive_Eval/omnimmi/data/asr_words/segments/<video>.json`. Each cache records `video`, source hashes, `transcription_scope`, and ordered `word/start/end` entries. No Whisper installation or ASR generation step is needed. PA and the visual-only condition skip ASR and the budget tokenizer entirely.

## Packaged annotations

- OVBench: `VLMEvalKit/json_data/ovbench/ovbench_split_with_fps_ava_raw_frame_processed_128_bbox1000.json` for the non-AVA rows; New AVA comes from the external BBox1000 JSON.
- ODVBench: `VLMEvalKit/json_data/odvbench/ODVbench_bbox1000.json`.
- OVO-Timing: `VLMEvalKit/json_data/ovobench/ovobench_timing.json`.
- ViSpeak: six files under `Proactive_Eval/ViSpeak-Bench/data/annotations/`.

Converted TSVs and media caches are created under the configured inference output root. Existing derived TSVs from other experiments are not required.

## Models

`run_script/configs/models.json` refers to `../path_to/models/OneStreamer-4B` and `../path_to/models/Qwen3-VL-4B-Instruct_no_special_token`. Each must be a complete Hugging Face checkpoint with configuration, tokenizer/processor files and safetensors weights (including all index-referenced shards). Do not add special tokens for Instruct. Both identities run the same prompts and settings.

ASR-enabled OmniMMI uses the Instruct tokenizer to enforce the 4096-token lexical budget for either model, so its tokenizer must also be available when evaluating OneStreamer-4B with ASR.

## Credentials and outputs

Credentials are optional for inference. For text judging, provide an API key environment variable or `JUDGE_ENV_FILE=../path_to/secrets/judge.env`. This file is external to the project. Default output roots are `outputs` and `judge_outputs`; runtime files can contain your resolved paths and are not release assets.
