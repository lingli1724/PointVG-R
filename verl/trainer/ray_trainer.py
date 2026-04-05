"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.rollout.multiturn.rollout_multiturn import RolloutMultiturn
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        if config.trainer.max_steps is not None:
            self.steps_per_epoch = len(train_dataloader) if len(train_dataloader) > 0 else None
        else:
            self.steps_per_epoch = (
                self.training_steps // config.trainer.total_epochs if config.trainer.total_epochs > 0 else None
            )

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        self._adv_importance_global_stat = float(config.algorithm.adv_importance_global_var_init)
        self._adv_importance_estimator_warned = False
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.n = self.config.worker.rollout.n
        if self.config.worker.rollout.multiturn:
            self.config.worker.rollout.n = 1
            self.config.worker.rollout.seed = None
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()
        if self.config.worker.rollout.multiturn:
            self.multiturn_rollout = RolloutMultiturn(
                actor_rollout_wg=self.actor_rollout_ref_wg,
                config=self.config.worker.rollout,
                tokenizer=self.tokenizer,
                processor=self.processor,
                n=self.n,
            )

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
            "adv_importance_global_stat": self._adv_importance_global_stat,
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
                self._adv_importance_global_stat = float(
                    tracker_info.get(
                        "adv_importance_global_stat",
                        tracker_info.get("adv_importance_global_var", self._adv_importance_global_stat),
                    )
                )
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    @staticmethod
    def _is_negative_sample_from_gt(gt_raw: Any) -> bool:
        if gt_raw is None:
            return True
        if isinstance(gt_raw, str):
            s = gt_raw.strip().lower()
            if s == "none":
                return True
            if not s:
                return False
            try:
                gt = json.loads(gt_raw)
            except Exception:
                return False
        elif isinstance(gt_raw, dict):
            gt = gt_raw
        else:
            return False
        return isinstance(gt, dict) and gt.get("stage") == 1 and gt.get("hand_bbox") is None

    def _compute_adv_importance_statistic(self, group_scores: torch.Tensor) -> torch.Tensor:
        cfg = self.config.algorithm
        statistic = str(cfg.adv_importance_statistic)
        if group_scores.numel() <= 1:
            return torch.zeros((), dtype=group_scores.dtype, device=group_scores.device)
        if statistic == "variance":
            return torch.var(group_scores, unbiased=False)
        if statistic == "std":
            return torch.std(group_scores, unbiased=False)
        if statistic == "top_bottom_gap":
            return torch.max(group_scores) - torch.min(group_scores)
        if statistic == "reward_entropy":
            score_min = torch.min(group_scores)
            score_max = torch.max(group_scores)
            if float((score_max - score_min).detach().item()) <= cfg.adv_importance_eps:
                return torch.zeros((), dtype=group_scores.dtype, device=group_scores.device)
            hist = torch.histc(
                group_scores.float(),
                bins=int(cfg.adv_importance_entropy_bins),
                min=float(score_min.detach().item()),
                max=float(score_max.detach().item()),
            )
            probs = hist / hist.sum().clamp_min(1.0)
            return -(probs * torch.log(probs + cfg.adv_importance_eps)).sum().to(group_scores.dtype)
        raise ValueError(f"Unsupported adv importance statistic: {statistic}")

    def _compute_adv_importance_weight(
        self,
        statistic_per_sample: torch.Tensor,
        current_batch_stat: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        cfg = self.config.algorithm
        eps = float(cfg.adv_importance_eps)
        global_baseline = max(eps, float(self._adv_importance_global_stat))
        batch_baseline = float(current_batch_stat.detach().item()) + eps
        use_batch_stat = bool(cfg.adv_importance_use_batch_stat)
        baseline_value = batch_baseline if use_batch_stat else global_baseline
        baseline_tensor = statistic_per_sample.new_tensor(baseline_value)

        if cfg.adv_importance_method == "sqrt_ratio":
            importance_weight = torch.sqrt(statistic_per_sample / baseline_tensor)
        elif cfg.adv_importance_method == "linear_ratio":
            importance_weight = statistic_per_sample / baseline_tensor
        elif cfg.adv_importance_method == "power_ratio":
            importance_weight = torch.pow(statistic_per_sample / baseline_tensor, float(cfg.adv_importance_power))
        elif cfg.adv_importance_method == "log_ratio":
            ratio = statistic_per_sample / baseline_tensor
            importance_weight = torch.log1p(float(cfg.adv_importance_log_alpha) * ratio)
        else:
            raise ValueError(f"Unsupported adv importance method: {cfg.adv_importance_method}")

        self._adv_importance_global_stat = max(
            eps,
            cfg.adv_importance_momentum * global_baseline
            + (1.0 - cfg.adv_importance_momentum) * float(current_batch_stat.detach().item()),
        )

        return importance_weight, float(baseline_value)

    def _maybe_apply_adv_importance_weighting(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        cfg = self.config.algorithm
        if not cfg.enable_adv_importance_weighting:
            return

        supported_estimators = (AdvantageEstimator.GRPO, AdvantageEstimator.GRPO_PASSK, AdvantageEstimator.RLOO)
        if cfg.adv_estimator not in supported_estimators:
            if not self._adv_importance_estimator_warned:
                print(
                    "Skip `algorithm.enable_adv_importance_weighting`: "
                    f"unsupported adv_estimator={cfg.adv_estimator}."
                )
                self._adv_importance_estimator_warned = True
            return

        if "advantages" not in batch.batch:
            return
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("`uid` is required for advantage importance weighting.")

        score_key = "token_level_scores" if "token_level_scores" in batch.batch else "token_level_rewards"
        seq_scores = batch.batch[score_key].sum(dim=-1)
        uids = batch.non_tensor_batch["uid"]

        uid2indices = defaultdict(list)
        for i, uid in enumerate(uids):
            uid2indices[uid].append(i)

        if len(uid2indices) == 0:
            return

        statistic_per_sample = torch.zeros_like(seq_scores)
        statistic_values = []
        for indices in uid2indices.values():
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=seq_scores.device)
            group_scores = seq_scores.index_select(0, indices_tensor)
            group_stat = self._compute_adv_importance_statistic(group_scores)
            statistic_per_sample[indices_tensor] = group_stat
            statistic_values.append(group_stat)

        current_batch_stat = torch.stack(statistic_values).mean()
        importance_weight, baseline_metric = self._compute_adv_importance_weight(
            statistic_per_sample=statistic_per_sample,
            current_batch_stat=current_batch_stat,
        )

        importance_weight = torch.clamp(
            importance_weight,
            min=cfg.adv_importance_clip_min,
            max=cfg.adv_importance_clip_max,
        )
        if cfg.adv_importance_normalize:
            importance_weight = importance_weight / (importance_weight.mean() + cfg.adv_importance_eps)

        batch.batch["advantages"] = batch.batch["advantages"] * importance_weight.unsqueeze(-1)

        metrics["algorithm/adv_importance_weight_mean"] = float(importance_weight.mean().detach().item())
        metrics["algorithm/adv_importance_weight_max"] = float(importance_weight.max().detach().item())
        metrics["algorithm/adv_importance_weight_min"] = float(importance_weight.min().detach().item())
        metrics["algorithm/adv_importance_group_stat"] = float(current_batch_stat.detach().item())
        metrics["algorithm/adv_importance_global_stat"] = float(baseline_metric)

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step, tag="val")

    def _format_reward_detail(self, reward_metrics: dict[str, list[float]], idx: int) -> str:
        preferred_keys = [
            "overall",
            "reward",
            "neg",
            "iou",
            "ray_cos",
            "kpt_mpjpe_score",
            "kpt_bbox_iou",
            "caption_sim",
            "caption_len_ratio",
            "format",
            "text_sim",
        ]
        all_keys = [k for k, v in reward_metrics.items() if v and idx < len(v)]
        extra_keys = sorted([k for k in all_keys if k not in preferred_keys])
        keys = [k for k in preferred_keys if k in all_keys] + extra_keys
        parts = []
        for key in keys:
            values = reward_metrics.get(key)
            if not values or idx >= len(values):
                continue
            val = values[idx]
            try:
                parts.append(f"{key}={float(val):.4f}")
            except Exception:
                parts.append(f"{key}={val}")
        return ", ".join(parts)

    def _get_logged_response_text(
        self,
        non_tensor_batch: dict[str, Any],
        idx: int,
        fallback_response_text: str,
    ) -> str:
        """Log reward-input text first, then optional multiturn debug text."""
        multiturn_response_text = None
        multiturn_full_text = None

        if "multiturn_response_text" in non_tensor_batch:
            multiturn_response_text = str(non_tensor_batch["multiturn_response_text"][idx])
        if "multiturn_full_text" in non_tensor_batch:
            multiturn_full_text = str(non_tensor_batch["multiturn_full_text"][idx])

        # `fallback_response_text` is decoded from `responses` tensor, which is exactly
        # what reward_function consumes. Keep it as the primary log body.
        blocks = [fallback_response_text]

        if multiturn_response_text is not None and multiturn_response_text != fallback_response_text:
            blocks.append(f"[multiturn_response_text]\n{multiturn_response_text}")

        if multiturn_full_text is not None and multiturn_full_text != fallback_response_text:
            blocks.append(f"[multiturn_full_text]\n{multiturn_full_text}")

        return "\n\n".join(blocks)

    def _maybe_log_train_generations(self, batch: DataProto, reward_metrics: dict[str, list[float]]) -> None:
        if self.config.trainer.train_generations_to_log <= 0:
            return
        if self.config.trainer.train_generations_log_freq > 0:
            if self.global_step % self.config.trainer.train_generations_log_freq != 0:
                return
        else:
            if not self.steps_per_epoch:
                return
            if self.global_step % self.steps_per_epoch != 0:
                return

        max_response_length = batch.batch["responses"].size(-1)
        prompt_ids = batch.batch["prompts"]
        attention_mask = batch.batch["attention_mask"]
        response_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        labels = batch.non_tensor_batch["ground_truth"]

        samples = []
        rng = np.random.RandomState(42 + self.global_step)
        indices = np.arange(len(batch))
        rng.shuffle(indices)
        indices = indices[: self.config.trainer.train_generations_to_log]

        for idx in indices:
            prompt_mask = attention_mask[idx][:-max_response_length].bool()
            prompt_tokens = prompt_ids[idx][prompt_mask]
            response_tokens = response_ids[idx][response_mask[idx].bool()]

            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            response_text = self._get_logged_response_text(
                non_tensor_batch=batch.non_tensor_batch,
                idx=idx,
                fallback_response_text=self.tokenizer.decode(response_tokens, skip_special_tokens=False),
            )

            label = labels[idx]
            if not isinstance(label, str):
                label = json.dumps(label, ensure_ascii=False)

            detail = self._format_reward_detail(reward_metrics, idx)
            if detail:
                response_text = f"{response_text}\n\n[reward_detail] {detail}"

            score = reward_metrics.get("overall", [0.0])[idx] if reward_metrics.get("overall") else 0.0
            samples.append((prompt_text, response_text, label, float(score)))

        self.logger.log_generation(samples, self.global_step, tag="train")

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            # Do not overwrite existing meta_info (e.g., eos_token_id); merge override on top.
            merged_meta_info = dict(test_gen_batch.meta_info)
            merged_meta_info.update(dict(self.config.worker.rollout.val_override_config))
            test_gen_batch.meta_info = merged_meta_info
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            if self.config.worker.rollout.multiturn:
                test_output_gen_batch = self.multiturn_rollout.generate_sequences(test_gen_batch)
            else:
                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            output_ids = test_batch.batch["responses"]
            attention_mask = test_batch.batch["attention_mask"]
            response_mask = test_batch.batch["response_mask"]
            max_response_length = output_ids.size(-1)
            input_texts = []
            output_texts = []
            for i in range(output_ids.size(0)):
                prompt_mask = attention_mask[i][:-max_response_length].bool()
                prompt_tokens = input_ids[i][prompt_mask]
                response_tokens = output_ids[i][response_mask[i].bool()]
                input_texts.append(self.tokenizer.decode(prompt_tokens, skip_special_tokens=False))
                response_text = self._get_logged_response_text(
                    non_tensor_batch=test_batch.non_tensor_batch,
                    idx=i,
                    fallback_response_text=self.tokenizer.decode(response_tokens, skip_special_tokens=False),
                )
                detail = self._format_reward_detail(reward_metrics, i)
                if detail:
                    response_text = f"{response_text}\n\n[reward_detail] {detail}"
                output_texts.append(response_text)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
            if self.config.worker.rollout.multiturn:
                gen_batch_output = self.multiturn_rollout.generate_sequences(gen_batch)
            else:
                gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=pad_size * self.n)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        reward_metrics_raw = reward_metrics
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)
                        self._maybe_log_train_generations(batch, reward_metrics_raw)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )
                    self._maybe_apply_adv_importance_weighting(batch=batch, metrics=metrics)

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
