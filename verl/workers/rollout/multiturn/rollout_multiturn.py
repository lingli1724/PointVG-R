
import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from ....protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ....utils import torch_functional as VF
from ....utils.dataset import process_image
from ...rollout.config import RolloutConfig


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"</answer>", re.IGNORECASE)
BOX_TAG_VALUE_RE = re.compile(
    r"<\|box_start\|>\s*"
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*"
    r"<\|box_end\|>"
)


def _parse_draw_ray_tool_strict(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text.strip())
    except Exception:
        return None

    # Keep strict behavior aligned with the reward parser.
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != {"name", "start", "end", "color"}:
        return None
    if obj.get("name") != "draw_ray":
        return None
    if obj.get("color") != "red":
        return None

    start = obj.get("start")
    end = obj.get("end")
    if (
        not isinstance(start, list)
        or not isinstance(end, list)
        or len(start) != 2
        or len(end) != 2
        or not all(isinstance(x, int) for x in start + end)
    ):
        return None

    # Keep safety check for draw backend.
    try:
        ImageColor.getcolor("red", "RGB")
    except Exception:
        return None

    return {"start": start, "end": end, "color": "red"}


def _extract_first_hand_bbox(text: str) -> Optional[List[int]]:
    m = BOX_TAG_VALUE_RE.search(text or "")
    if m is None:
        return None
    return [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]


def _ensure_pil(image_obj: Any, min_pixels: Optional[int], max_pixels: Optional[int]) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        img = image_obj
    else:
        img = process_image(image_obj, min_pixels=min_pixels, max_pixels=max_pixels)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _draw_ray_image(
    image_obj: Any,
    start: List[int],
    end: List[int],
    color: str,
    width: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> Image.Image:
    base = _ensure_pil(image_obj, min_pixels=min_pixels, max_pixels=max_pixels)
    out = base.copy()
    draw = ImageDraw.Draw(out)
    draw.line([(start[0], start[1]), (end[0], end[1])], fill=color, width=width)
    return out


class RolloutMultiturn:
    def __init__(
        self,
        actor_rollout_wg,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        n: int = 1,
    ):
        self.rollout = actor_rollout_wg
        self.cfg = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_id = tokenizer.pad_token_id
        self.n = n
        self.vocab_size = len(tokenizer)

    def _strip_pad_tokens(self, tokens: List[int]) -> List[int]:
        tokens = [t for t in tokens if t != self.pad_id and t < self.vocab_size]
        blocked = set()
        for attr in ("image_token", "vision_start_token", "vision_end_token", "video_token"):
            tok = getattr(self.processor, attr, None) if self.processor is not None else None
            if isinstance(tok, str):
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tok_id, int) and tok_id >= 0:
                    blocked.add(tok_id)
        return [t for t in tokens if t not in blocked]

    def _append_observation(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
        images: List[Any],
        ray: Dict[str, Any],
        hand_bbox: Optional[List[int]],
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ) -> Tuple[List[int], List[int], List[Any]]:
        if len(images) == 0:
            return prompt_ids, response_ids, images

        try:
            ray_img = _draw_ray_image(
                image_obj=images[0],
                start=ray["start"],
                end=ray["end"],
                color=ray["color"],
                width=self.cfg.ray_line_width,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        except Exception:
            # Treat draw failures as invalid tool observations; caller will stop this sample safely.
            return prompt_ids, response_ids, images

        obs_payload = {
            "hand_bbox": hand_bbox,
            "ray": {
                "start": ray["start"],
                "end": ray["end"],
            },
        }
        obs_txt = (
            "\n<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "<image>\n"
            f"I have drawn the ray on the image as requested : {json.dumps(obs_payload, ensure_ascii=False)}. "
            "Please continue your analysis.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        new_ids = self.tokenizer.encode(obs_txt, add_special_tokens=False)
        new_ids = [token_id for token_id in new_ids if token_id < self.vocab_size]

        # Grounded-rl style safety guard:
        # image tokens introduced by a new observation image should be budgeted against response_length,
        # otherwise later left-truncation can break image-token/feature alignment in training.
        image_tokens = int((ray_img.size[0] * ray_img.size[1]) / (28 * 28))
        if len(response_ids) + len(new_ids) + image_tokens >= self.cfg.response_length:
            return prompt_ids, response_ids, images

        prompt_ids.extend(new_ids)
        response_ids.extend(new_ids)
        images.append(ray_img)
        return prompt_ids, response_ids, images

    def _build_position_ids(self, obs_inputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        if self.processor is None:
            return torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ....models.transformers.qwen3_vl import get_rope_index
            else:
                from ....models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=obs_inputs["input_ids"][0],
                image_grid_thw=obs_inputs.get("image_grid_thw", None),
                video_grid_thw=obs_inputs.get("video_grid_thw", None),
                second_per_grid_ts=obs_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(obs_inputs["input_ids"][0])).unsqueeze(0)
            return torch.cat((text_position_ids, vision_position_ids), dim=0)

        return torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        # Validation/meta override may not carry eos_token_id; fallback to tokenizer to avoid hard failure.
        eos_token_id: int = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)

        batch_size = input_ids.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        raw_prompt_ids = list(non_tensor_batch["raw_prompt_ids"])
        multi_modal_data = list(non_tensor_batch.get("multi_modal_data", np.array([{}] * batch_size, dtype=object)))

        n = int(prompts.meta_info.get("n", self.n))
        if n > 1:
            input_ids = input_ids.repeat_interleave(n, dim=0)
            attention_mask = attention_mask.repeat_interleave(n, dim=0)
            position_ids = position_ids.repeat_interleave(n, dim=0)
            raw_prompt_ids = [copy.deepcopy(x) for x in raw_prompt_ids for _ in range(n)]
            multi_modal_data = [copy.deepcopy(x) for x in multi_modal_data for _ in range(n)]
            batch_size = input_ids.size(0)

        prompt_ids = [list(ids) for ids in raw_prompt_ids]
        image_buffers: List[List[Any]] = []
        for md in multi_modal_data:
            images = md.get("images", []) if isinstance(md, dict) else []
            image_buffers.append(list(images))

        min_pixels = prompts.meta_info.get("min_pixels")
        max_pixels = prompts.meta_info.get("max_pixels")

        finished = [False] * batch_size
        response_tokens: List[List[int]] = [[] for _ in range(batch_size)]

        for _ in range(self.cfg.max_iterations):
            active_idx = [i for i, done in enumerate(finished) if not done]
            if not active_idx:
                break

            active_dp = DataProto(
                batch=TensorDict(
                    {
                        "input_ids": input_ids[active_idx],
                        "attention_mask": attention_mask[active_idx],
                        "position_ids": position_ids[active_idx],
                    },
                    batch_size=(len(active_idx),),
                ),
                non_tensor_batch={
                    "raw_prompt_ids": np.array([prompt_ids[i] for i in active_idx], dtype=object),
                    "multi_modal_data": np.array([{"images": image_buffers[i]} for i in active_idx], dtype=object),
                },
                meta_info=dict(prompts.meta_info),
            )

            # Similar to grounded-rl multiturn handling: active subset size changes by turn,
            # so we pad/unpad per turn to satisfy equal-chunk dispatch.
            world_size = int(getattr(self.rollout, "world_size", 1))
            pad_size = 0
            if world_size > 1:
                active_dp, pad_size = pad_dataproto_to_divisor(active_dp, world_size)

            out_dp = self.rollout.generate_sequences(active_dp)
            if pad_size > 0:
                out_dp = unpad_dataproto(out_dp, pad_size=pad_size)
            out_ids = out_dp.batch["responses"].cpu().tolist()

            for local_i, global_i in enumerate(active_idx):
                toks = self._strip_pad_tokens(out_ids[local_i])
                if not toks:
                    finished[global_i] = True
                    continue

                # Hard cap per-sample response budget to avoid post-hoc truncation inconsistencies.
                remain = self.cfg.response_length - len(response_tokens[global_i])
                if remain <= 0:
                    finished[global_i] = True
                    continue
                if len(toks) > remain:
                    toks = toks[:remain]
                    finished[global_i] = True

                response_tokens[global_i].extend(toks)
                prompt_ids[global_i].extend(toks)
                text = self.tokenizer.decode(toks, skip_special_tokens=False)

                if ANSWER_RE.search(text):
                    finished[global_i] = True
                    continue

                if self.cfg.limit_images > 0 and len(image_buffers[global_i]) >= self.cfg.limit_images:
                    finished[global_i] = True
                    continue

                m = TOOL_RE.search(text)
                if m is None:
                    finished[global_i] = True
                    continue

                ray = _parse_draw_ray_tool_strict(m.group(1))
                if ray is None:
                    finished[global_i] = True
                    continue

                hand_bbox = _extract_first_hand_bbox(text)
                old_image_count = len(image_buffers[global_i])
                prompt_ids[global_i], response_tokens[global_i], image_buffers[global_i] = self._append_observation(
                    prompt_ids[global_i],
                    response_tokens[global_i],
                    image_buffers[global_i],
                    ray,
                    hand_bbox,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                # If observation is skipped due budget guard, stop this sample to avoid invalid MM alignment.
                if len(image_buffers[global_i]) == old_image_count:
                    finished[global_i] = True

        max_total_len = self.cfg.prompt_length + self.cfg.response_length

        seq_ids_list = []
        attn_mask_list = []
        pos_ids_list = []
        response_mask_list = []
        multi_modal_inputs_list: List[Dict[str, torch.Tensor]] = []

        for i in range(batch_size):
            images = [
                _ensure_pil(img, min_pixels=min_pixels, max_pixels=max_pixels) for img in image_buffers[i]
            ]
            full_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=False)

            obs_inputs = self.processor(
                images if len(images) > 0 else None,
                [full_text],
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=max_total_len,
            )
            mm_inputs = {}
            for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
                if key in obs_inputs:
                    mm_inputs[key] = obs_inputs[key]
            multi_modal_inputs_list.append(mm_inputs)
            ids = obs_inputs["input_ids"][0]
            attn = obs_inputs["attention_mask"][0]
            pos = self._build_position_ids(obs_inputs, attn)

            ids, attn, pos = VF.postprocess_data(
                input_ids=ids,
                attention_mask=attn,
                position_ids=pos,
                max_length=max_total_len,
                pad_token_id=self.pad_id,
                left_pad=True,
                truncation="left",
            )

            num_input_tokens = int(attention_mask[i].sum().item())
            mask_full = torch.zeros_like(attn)
            if num_input_tokens < attn.size(0):
                mask_full[num_input_tokens:] = attn[num_input_tokens:]

            seq_ids_list.append(ids.tolist())
            attn_mask_list.append(attn.tolist())
            if pos.dim() == 2:
                pos_ids_list.append(pos.tolist())
            else:
                pos_ids_list.append(pos.tolist())
            response_mask_list.append(mask_full[-self.cfg.response_length :].tolist())

        seq_ids = VF.pad_2d_list_to_length(seq_ids_list, self.pad_id, max_length=max_total_len).to(input_ids.device)
        attn_mask = VF.pad_2d_list_to_length(attn_mask_list, 0, max_length=max_total_len).to(input_ids.device)
        response_mask = VF.pad_2d_list_to_length(
            response_mask_list, 0, max_length=self.cfg.response_length
        ).to(input_ids.device)

        if isinstance(pos_ids_list[0][0], list):
            pos_tensors = [torch.tensor(x, dtype=position_ids.dtype) for x in pos_ids_list]
            pos_ids = torch.stack(pos_tensors, dim=0).to(input_ids.device)
        else:
            pos_ids = VF.pad_2d_list_to_length(pos_ids_list, 0, max_length=max_total_len).to(input_ids.device)

        responses = seq_ids[:, -self.cfg.response_length :]
        prompts_out = input_ids[:, -self.cfg.prompt_length :]

        batch = TensorDict(
            {
                "prompts": prompts_out,
                "responses": responses,
                "input_ids": seq_ids,
                "attention_mask": attn_mask,
                "response_mask": response_mask,
                "position_ids": pos_ids,
            },
            batch_size=batch_size,
        )

        non_tensor_out = {
            "multi_modal_data": np.array([{"images": imgs} for imgs in image_buffers], dtype=object),
            # Keep multimodal processor outputs aligned with rebuilt input_ids to avoid token/feature drift.
            "multi_modal_inputs": np.array(multi_modal_inputs_list, dtype=object),
            # Keep full multiturn transcript for logging/debugging.
            "multiturn_full_text": np.array(
                [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in prompt_ids], dtype=object
            ),
            # Keep generated span after initial prompt boundary.
            "multiturn_response_text": np.array(
                [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in response_tokens], dtype=object
            ),
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_out, meta_info=prompts.meta_info)
