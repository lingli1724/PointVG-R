#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL_PATH=.../path/to/model
TRAIN_FILES=.../path/to/train_data.jsonl
VAL_FILES=.../path/to/val_data.jsonl
REWARD_FUNC="${SCRIPT_DIR}/reward_function/reward_func.py:compute_score"
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
EXPERIMENT_NAME=opensource_exp
PROJECT_NAME=PointVG-R

LR=1e-5
WEIGHT_DECAY=1e-2
LR_SCHEDULER_TYPE=cosine
LR_WARMUP_RATIO=0.10
MIN_LR_RATIO=0.20
EPOCH=7
MAX_GRAD_NORM=0.5
KL_COEF=5.0e-2
KL_TYPE=adaptive
KL_TARGET=0.08
KL_HORIZON=10000
ROLLOUT_TEMPERATURE=0.9
ROLLOUT_TOP_P=0.95

MAX_PROMPT_LEN=1024
MAX_RESPONSE_LEN=1024
MAX_MODEL_LEN=4096
MAX_PIXELS=1048576

# Core ablation switches
ADV_STAT=variance
ADV_METHOD=sqrt_ratio
ADV_POWER=0.5
ADV_LOG_ALPHA=1.0
ADV_ENTROPY_BINS=5
ADV_CLIP_MIN=0.3
ADV_CLIP_MAX=3.0
ADV_NORMALIZE=false
ADV_GLOBAL_INIT=0.1
ADV_MOMENTUM=0.95

VLLM_USE_V1=1 python3 -m verl.trainer.main \
  trainer.total_epochs=${EPOCH} \
  trainer.project_name=${PROJECT_NAME} \
  data.train_files=${TRAIN_FILES} \
  data.val_files=${VAL_FILES} \
  config="${CONFIG_PATH}" \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=8 \
  trainer.save_limit=1 \
  trainer.save_freq=2 \
  trainer.val_freq=1 \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.actor.model.lora.rank=0 \
  worker.actor.global_batch_size=128 \
  worker.actor.optim.lr=${LR} \
  worker.actor.optim.weight_decay=${WEIGHT_DECAY} \
  worker.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE} \
  worker.actor.optim.lr_warmup_ratio=${LR_WARMUP_RATIO} \
  worker.actor.optim.min_lr_ratio=${MIN_LR_RATIO} \
  worker.actor.max_grad_norm=${MAX_GRAD_NORM} \
  worker.actor.micro_batch_size_per_device_for_update=8 \
  worker.actor.micro_batch_size_per_device_for_experience=16 \
  algorithm.kl_coef=${KL_COEF} \
  algorithm.kl_type=${KL_TYPE} \
  algorithm.kl_target=${KL_TARGET} \
  algorithm.kl_horizon=${KL_HORIZON} \
  worker.rollout.temperature=${ROLLOUT_TEMPERATURE} \
  worker.rollout.top_p=${ROLLOUT_TOP_P} \
  data.max_prompt_length=${MAX_PROMPT_LEN} \
  data.max_response_length=${MAX_RESPONSE_LEN} \
  data.max_pixels=${MAX_PIXELS} \
  worker.rollout.multiturn=true \
  worker.rollout.limit_images=2 \
  worker.rollout.max_iterations=2 \
  worker.rollout.max_model_len=${MAX_MODEL_LEN} \
  worker.rollout.max_generation_length_per_turn=${MAX_RESPONSE_LEN} \
  worker.rollout.stop_strings="</tool_call>,</answer>,<|im_end|>" \
  worker.rollout.ray_line_width=5 \
  worker.reward.reward_function=${REWARD_FUNC} \
  algorithm.enable_adv_importance_weighting=true \
  algorithm.adv_importance_statistic=${ADV_STAT} \
  algorithm.adv_importance_method=${ADV_METHOD} \
  algorithm.adv_importance_power=${ADV_POWER} \
  algorithm.adv_importance_log_alpha=${ADV_LOG_ALPHA} \
  algorithm.adv_importance_entropy_bins=${ADV_ENTROPY_BINS} \
  algorithm.adv_importance_global_var_init=${ADV_GLOBAL_INIT} \
  algorithm.adv_importance_momentum=${ADV_MOMENTUM} \
  algorithm.adv_importance_eps=1e-6 \
  algorithm.adv_importance_clip_min=${ADV_CLIP_MIN} \
  algorithm.adv_importance_clip_max=${ADV_CLIP_MAX} \
  algorithm.adv_importance_normalize=${ADV_NORMALIZE} \
  trainer.find_last_checkpoint=true
