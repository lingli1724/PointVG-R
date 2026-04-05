# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# PPO config
# """

# import os
# from dataclasses import asdict, dataclass, field, fields, is_dataclass
# from typing import Optional, Tuple

# from ..utils.py_functional import get_abs_path
# from ..workers.config import WorkerConfig


# def recursive_post_init(dataclass_obj):
#     if hasattr(dataclass_obj, "post_init"):
#         dataclass_obj.post_init()

#     for attr in fields(dataclass_obj):
#         if is_dataclass(getattr(dataclass_obj, attr.name)):
#             recursive_post_init(getattr(dataclass_obj, attr.name))


# @dataclass
# class DataConfig:
#     train_files: str = ""
#     val_files: str = ""
#     prompt_key: str = "prompt"
#     answer_key: str = "answer"
#     image_key: str = "images"
#     video_key: str = "videos"
#     image_dir: Optional[str] = None
#     video_fps: float = 2.0
#     max_prompt_length: int = 512
#     max_response_length: int = 512
#     rollout_batch_size: int = 512
#     mini_rollout_batch_size: Optional[int] = None
#     val_batch_size: int = -1
#     format_prompt: Optional[str] = None
#     override_chat_template: Optional[str] = None
#     shuffle: bool = True
#     seed: int = 1
#     min_pixels: Optional[int] = 262144
#     max_pixels: Optional[int] = 4194304
#     filter_overlong_prompts: bool = True
#     filter_overlong_prompts_workers: int = 16

#     def post_init(self):
#         self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
#         self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
#         self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


# @dataclass
# class AlgorithmConfig:
#     gamma: float = 1.0
#     """discount factor for ppo gae advantage estimator"""
#     lam: float = 1.0
#     """lambda value for ppo gae advantage estimator"""
#     adv_estimator: str = "grpo"
#     """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
#     disable_kl: bool = False
#     """disable reference model"""
#     use_kl_loss: bool = False
#     """use kl loss instead of kl in reward"""
#     kl_penalty: str = "kl"
#     """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
#     kl_coef: float = 1e-3
#     """kl coefficient"""
#     kl_type: str = "fixed"
#     """kl controller type, support `fixed`, `adaptive`"""
#     kl_horizon: float = 10000.0
#     """kl horizon for adaptive kl controller"""
#     kl_target: float = 0.1
#     """target kl for adaptive kl controller"""
#     online_filtering: bool = False
#     """use online filtering"""
#     filter_key: str = "overall"
#     """reward key for filtering samples"""
#     filter_low: float = 0.01
#     """filter out low reward samples if online filtering"""
#     filter_high: float = 0.99
#     """filter out high reward samples if online filtering"""
#     enable_adv_importance_weighting: bool = False
#     """enable dynamic importance weighting on advantage using per-group score variance"""
#     adv_importance_global_var_init: float = 1.0
#     """initial global variance for dynamic importance weighting"""
#     adv_importance_momentum: float = 0.95
#     """momentum for updating global variance, new = m * old + (1 - m) * current"""
#     adv_importance_eps: float = 1e-6
#     """epsilon for numerical stability"""
#     adv_importance_clip_min: float = 0.1
#     """minimum clip value for importance weight"""
#     adv_importance_clip_max: float = 10.0
#     """maximum clip value for importance weight"""
#     adv_importance_normalize: bool = False
#     """normalize clipped importance weights by batch mean to keep mean close to 1"""
#     adv_importance_statistic: str = "variance"
#     """group-level statistic used to compute importance: variance, std, reward_entropy, top_bottom_gap"""
#     adv_importance_method: str = "original"
#     """importance weight mapping: original, sqrt_ratio, linear_ratio, power_ratio, log_ratio"""
#     adv_importance_power: float = 0.5
#     """exponent used by power_ratio"""
#     adv_importance_log_alpha: float = 1.0
#     """scale factor used by log_ratio"""
#     adv_importance_entropy_bins: int = 5
#     """number of bins for reward entropy statistic"""

#     def post_init(self):
#         if self.adv_importance_global_var_init <= 0:
#             raise ValueError("`algorithm.adv_importance_global_var_init` must be > 0.")
#         if not (0.0 <= self.adv_importance_momentum < 1.0):
#             raise ValueError("`algorithm.adv_importance_momentum` must be in [0, 1).")
#         if self.adv_importance_eps <= 0:
#             raise ValueError("`algorithm.adv_importance_eps` must be > 0.")
#         if self.adv_importance_clip_min <= 0 or self.adv_importance_clip_max <= 0:
#             raise ValueError("`algorithm.adv_importance_clip_min/max` must be > 0.")
#         if self.adv_importance_clip_min > self.adv_importance_clip_max:
#             raise ValueError("`algorithm.adv_importance_clip_min` must be <= `algorithm.adv_importance_clip_max`.")
#         if self.adv_importance_statistic not in ("variance", "std", "reward_entropy", "top_bottom_gap"):
#             raise ValueError(
#                 "`algorithm.adv_importance_statistic` must be one of: variance, std, reward_entropy, top_bottom_gap."
#             )
#         if self.adv_importance_method not in ("original", "sqrt_ratio", "linear_ratio", "power_ratio", "log_ratio"):
#             raise ValueError(
#                 "`algorithm.adv_importance_method` must be one of: "
#                 "original, sqrt_ratio, linear_ratio, power_ratio, log_ratio."
#             )
#         if self.adv_importance_power <= 0:
#             raise ValueError("`algorithm.adv_importance_power` must be > 0.")
#         if self.adv_importance_log_alpha <= 0:
#             raise ValueError("`algorithm.adv_importance_log_alpha` must be > 0.")
#         if self.adv_importance_entropy_bins < 2:
#             raise ValueError("`algorithm.adv_importance_entropy_bins` must be >= 2.")


# @dataclass
# class TrainerConfig:
#     total_epochs: int = 15
#     """total epochs for training"""
#     max_steps: Optional[int] = None
#     """max steps for training, if specified, total_epochs is ignored"""
#     project_name: str = "easy_r1"
#     """project name for logger"""
#     experiment_name: str = "demo"
#     """experiment name for logger"""
#     logger: Tuple[str] = ("console", "wandb")
#     """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
#     nnodes: int = 1
#     """number of nodes for training"""
#     n_gpus_per_node: int = 8
#     """number of gpus per node for training"""
#     max_try_make_batch: int = 20
#     """max number of generations for online filtering, -1 means no limit"""
#     critic_warmup: int = 0
#     """critic warmup steps"""
#     val_freq: int = -1
#     """validation frequency, -1 means no validation"""
#     val_before_train: bool = True
#     """validate before training"""
#     val_only: bool = False
#     """validate only, skip training"""
#     val_generations_to_log: int = 0
#     """number of generations to log for validation"""
#     train_generations_to_log: int = 0
#     """number of generations to log for training (logged once per epoch)"""
#     train_generations_log_freq: int = 0
#     """log training generations every N steps (0 means disable, 1 means every step)"""
#     save_freq: int = -1
#     """save frequency, -1 means no saving"""
#     save_limit: int = -1
#     """max number of checkpoints to save, -1 means no limit"""
#     save_model_only: bool = False
#     """save model only, no optimizer state dict"""
#     save_checkpoint_path: Optional[str] = None
#     """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
#     load_checkpoint_path: Optional[str] = None
#     """load checkpoint path"""
#     ray_timeline: Optional[str] = None
#     """file to save ray timeline"""
#     find_last_checkpoint: bool = True
#     """automatically find the last checkpoint in the save checkpoint path to resume training"""

#     def post_init(self):
#         if self.save_checkpoint_path is None:
#             self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

#         self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
#         self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


# @dataclass
# class PPOConfig:
#     data: DataConfig = field(default_factory=DataConfig)
#     worker: WorkerConfig = field(default_factory=WorkerConfig)
#     algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
#     trainer: TrainerConfig = field(default_factory=TrainerConfig)

#     def post_init(self):
#         self.worker.rollout.prompt_length = self.data.max_prompt_length
#         self.worker.rollout.response_length = self.data.max_response_length
#         self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
#         self.worker.actor.disable_kl = self.algorithm.disable_kl
#         self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
#         self.worker.actor.kl_penalty = self.algorithm.kl_penalty
#         self.worker.actor.kl_coef = self.algorithm.kl_coef

#     def deep_post_init(self):
#         recursive_post_init(self)

#     def to_dict(self):
#         return asdict(self)
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    enable_adv_importance_weighting: bool = False
    """enable dynamic importance weighting on advantage using per-group score variance"""
    adv_importance_global_var_init: float = 1.0
    """initial global variance for dynamic importance weighting"""
    adv_importance_momentum: float = 0.95
    """momentum for updating global variance, new = m * old + (1 - m) * current"""
    adv_importance_eps: float = 1e-6
    """epsilon for numerical stability"""
    adv_importance_clip_min: float = 0.1
    """minimum clip value for importance weight"""
    adv_importance_clip_max: float = 10.0
    """maximum clip value for importance weight"""
    adv_importance_normalize: bool = False
    """normalize clipped importance weights by batch mean to keep mean close to 1"""
    adv_importance_statistic: str = "variance"
    """group-level statistic used to compute importance: variance, std, reward_entropy, top_bottom_gap"""
    adv_importance_method: str = "sqrt_ratio"
    """importance weight mapping: sqrt_ratio, linear_ratio, power_ratio, log_ratio"""
    adv_importance_use_batch_stat: bool = False
    """use current batch statistic as denominator instead of the EMA global statistic"""
    adv_importance_power: float = 0.5
    """exponent used by power_ratio"""
    adv_importance_log_alpha: float = 1.0
    """scale factor used by log_ratio"""
    adv_importance_entropy_bins: int = 5
    """number of bins for reward entropy statistic"""

    def post_init(self):
        if self.adv_importance_global_var_init <= 0:
            raise ValueError("`algorithm.adv_importance_global_var_init` must be > 0.")
        if not (0.0 <= self.adv_importance_momentum < 1.0):
            raise ValueError("`algorithm.adv_importance_momentum` must be in [0, 1).")
        if self.adv_importance_eps <= 0:
            raise ValueError("`algorithm.adv_importance_eps` must be > 0.")
        if self.adv_importance_clip_min <= 0 or self.adv_importance_clip_max <= 0:
            raise ValueError("`algorithm.adv_importance_clip_min/max` must be > 0.")
        if self.adv_importance_clip_min > self.adv_importance_clip_max:
            raise ValueError("`algorithm.adv_importance_clip_min` must be <= `algorithm.adv_importance_clip_max`.")
        if self.adv_importance_statistic not in ("variance", "std", "reward_entropy", "top_bottom_gap"):
            raise ValueError(
                "`algorithm.adv_importance_statistic` must be one of: variance, std, reward_entropy, top_bottom_gap."
            )
        if self.adv_importance_method not in ("sqrt_ratio", "linear_ratio", "power_ratio", "log_ratio"):
            raise ValueError(
                "`algorithm.adv_importance_method` must be one of: "
                "sqrt_ratio, linear_ratio, power_ratio, log_ratio."
            )
        if self.adv_importance_power <= 0:
            raise ValueError("`algorithm.adv_importance_power` must be > 0.")
        if self.adv_importance_log_alpha <= 0:
            raise ValueError("`algorithm.adv_importance_log_alpha` must be > 0.")
        if self.adv_importance_entropy_bins < 2:
            raise ValueError("`algorithm.adv_importance_entropy_bins` must be >= 2.")


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    train_generations_to_log: int = 0
    """number of generations to log for training (logged once per epoch)"""
    train_generations_log_freq: int = 0
    """log training generations every N steps (0 means disable, 1 means every step)"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
