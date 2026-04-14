# PointVG-R

## Overview

`PointVG-R` is a reinforcement learning training project for visual pointing understanding, built on top of the in-repo `verl` framework. The current codebase focuses on:

- Multi-GPU PPO/GRPO training with `Ray + veRL/FSDP + vLLM`
- Multi-modal inputs with text, image, and video fields
- A custom reward function that jointly scores hand boxes, pointing rays, keypoints, and target object boxes
- An open-source training entry script at [`PointVG-R/train.sh`](/home/zhanggl/liling_new/EgoHand_v1/code_opensource/PointVG-R/PointVG-R/train.sh) and a base config at [`PointVG-R/config.yaml`](/home/zhanggl/liling_new/EgoHand_v1/code_opensource/PointVG-R/PointVG-R/config.yaml)

## Repository Structure

```text
PointVG-R/
├── PointVG-R/
│   ├── config.yaml
│   ├── train.sh
│   └── reward_function/
│       └── reward_func.py
├── dataset/
├── scripts/
├── tests/
├── verl/
└── pyproject.toml
```

What each part is for:

- `PointVG-R/train.sh`: training launcher with commonly used hyperparameters and command-line overrides
- `PointVG-R/config.yaml`: default training config covering data fields, algorithm settings, worker settings, and trainer settings
- `PointVG-R/reward_function/reward_func.py`: custom reward function entrypoint `compute_score`
- `verl/`: core training framework, including data loading, trainer, actor/rollout/reward modules
- `dataset/`: recommended location for training/validation data or processed data artifacts
- `scripts/`: auxiliary scripts such as model merging
- `tests/`: partial unit tests

## Data Format

The dataset is loaded through `RLHFDataset` in [`verl/utils/dataset.py`](/home/zhanggl/liling_new/EgoHand_v1/code_opensource/PointVG-R/verl/utils/dataset.py). The current config uses these key fields:

- `prompt_key: prompt`
- `answer_key: ground_truth`
- `image_key: images`
- `video_key: videos`

That means each sample should contain at least:

- `prompt`: input text for the model
- `ground_truth`: annotation payload consumed by the reward function
- `images` or `videos`: optional multi-modal input lists

A recommended `jsonl` example:

```json
{"prompt":"Please identify the object being pointed at in the image.<image>","ground_truth":"{\"hand_bbox\":[10,20,100,120],\"pointing_ray\":{\"start\":[40,60],\"end\":[180,200]},\"pointing_keypoints\":[[40,60],[180,200]],\"obj_bbox\":[150,170,260,320]}","images":["example.jpg"]}
```

Notes:

- `ground_truth` is parsed by the reward function as either a JSON string or a dictionary.
- If `image_dir` is configured, relative paths in `images` or `videos` are joined with that directory.
- If the prompt contains `<image>` or `<video>`, the dataloader converts it into a multi-modal chat-template input.
- In the dataloader, the answer field is normalized into the `ground_truth` field used during training.

## Reward Function

The reward function is defined in [`PointVG-R/reward_function/reward_func.py`](/home/zhanggl/liling_new/EgoHand_v1/code_opensource/PointVG-R/PointVG-R/reward_function/reward_func.py), with the entrypoint:

```python
compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]
```

The current reward design includes:

- `hand_iou`: IoU between the predicted hand box and the ground-truth hand box
- `ray_cos`: directional consistency between the predicted pointing ray and the ground-truth ray
- `kpt_score`: normalized keypoint distance score
- `obj_iou`: IoU between the predicted target object box and the ground-truth object box
- `stage2_format`: whether the output format satisfies the stage-2 format constraints

The overall score is roughly:

```text
base = hand_iou + ray_cos + kpt_score + obj_iou * 5 + stage2_format * 2
reward = clamp(base * tool_penalty * bbox_penalty, 0, 10)
```

Where:

- repeated `draw_ray` tool calls reduce the score through `tool_penalty`
- multiple object boxes emitted after the last tool call reduce the score through `bbox_penalty`
- negative samples follow a separate scoring path in the code

## Training Configuration

### Default Config

The default config is defined in [`PointVG-R/config.yaml`](/home/zhanggl/liling_new/EgoHand_v1/code_opensource/PointVG-R/PointVG-R/config.yaml). Key fields include:

- Data:
  - `data.train_files`
  - `data.val_files`
  - `data.max_prompt_length`
  - `data.max_response_length`
  - `data.max_pixels`
- Model:
  - `worker.actor.model.model_path`
  - `worker.actor.model.trust_remote_code`
  - `worker.actor.model.lora.rank`
- Rollout:
  - `worker.rollout.n`
  - `worker.rollout.temperature`
  - `worker.rollout.top_p`
- Trainer:
  - `trainer.total_epochs`
  - `trainer.n_gpus_per_node`
  - `trainer.save_freq`
  - `trainer.val_freq`

### Training Script

The open-source training script is located at [`PointVG-R/train.sh`](/home/zhanggl/liling_new/EgoHand_v1/code_opensource/PointVG-R/PointVG-R/train.sh). Before launching, replace these paths:

- `MODEL_PATH`
- `TRAIN_FILES`
- `VAL_FILES`

The main training entry looks like:

```bash
bash PointVG-R/train.sh
```

Internally, the script runs:

```bash
VLLM_USE_V1=1 python3 -m verl.trainer.main ...
```

It also overrides:

- training epochs, learning rate, and weight decay
- KL control settings
- rollout temperature and top-p
- prompt and response lengths
- multi-turn rollout settings
- reward function path
- advantage importance weighting experiment settings
