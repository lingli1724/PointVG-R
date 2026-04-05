# reward func v6, 修改 ray cos score

import json
import math
import re
from typing import Any, Dict, List, Optional

REWARD_NAME = "egopoint_iou"
REWARD_TYPE = "batch"

CAPTION_RE = re.compile(r"<\|caption_start\|>(.*?)<\|caption_end\|>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
BOX_TAG_RE = re.compile(
    r"<\|box_start\|>\s*\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*<\|box_end\|>"
)
BOX_TAG_COORD_RE = re.compile(
    r"<\|box_start\|>\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*<\|box_end\|>"
)
POINT_TAG_RE = re.compile(r"<\|point_start\|>\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]\s*<\|point_end\|>")
POINT_TAG_COORD_RE = re.compile(
    r"<\|point_start\|>\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*<\|point_end\|>"
)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))


def _safe_json_load(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip() and x.strip().lower() != "none":
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _gt_is_none_like(gt_raw: Any) -> bool:
    if gt_raw is None:
        return True
    if isinstance(gt_raw, str):
        s = gt_raw.strip().lower()
        if s == "none":
            return True
        gt = _safe_json_load(gt_raw)
    elif isinstance(gt_raw, dict):
        gt = gt_raw
    else:
        return False
    return isinstance(gt, dict) and gt.get("stage") == 1 and gt.get("hand_bbox") is None


def _extract_tagged_boxes(text: str) -> List[List[int]]:
    return [list(map(int, m.groups())) for m in BOX_TAG_COORD_RE.finditer(text or "")]


def _pick_pred_box(response: str, first: bool) -> Optional[List[int]]:
    boxes = _extract_tagged_boxes(response)
    if not boxes:
        return None
    return boxes[0] if first else boxes[-1]


def _iou(a: Optional[List[int]], b: Any) -> float:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == 4 and len(b) == 4):
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else float(inter / union)


def _parse_draw_ray(obj: Any) -> Optional[Dict[str, List[int]]]:
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != {"name", "start", "end", "color"}:
        return None
    if obj.get("name") != "draw_ray" or obj.get("color") != "red":
        return None
    s, e = obj.get("start"), obj.get("end")
    if not (isinstance(s, list) and len(s) == 2 and isinstance(e, list) and len(e) == 2):
        return None
    try:
        return {"start": [int(s[0]), int(s[1])], "end": [int(e[0]), int(e[1])]}
    except Exception:
        return None


def _extract_ray(response: str) -> Optional[Dict[str, List[int]]]:
    for m in TOOL_CALL_RE.finditer(response or ""):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        ray = _parse_draw_ray(obj)
        if ray is not None:
            return ray
    return None


def _ray_cos(gt_ray: Any, pred_ray: Optional[Dict[str, List[int]]]) -> float:
    if not (isinstance(gt_ray, dict) and isinstance(pred_ray, dict)):
        return 0.0
    if "start" not in gt_ray or "end" not in gt_ray:
        return 0.0
    try:
        ax, ay = gt_ray["start"]
        bx, by = gt_ray["end"]
        cx, cy = pred_ray["start"]
        dx, dy = pred_ray["end"]
        v1x, v1y = float(bx - ax), float(by - ay)
        v2x, v2y = float(dx - cx), float(dy - cy)
        n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
        if n1 <= 1e-8 or n2 <= 1e-8:
            return 0.0
        cos_theta = _clamp((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0)
        angle = math.acos(cos_theta)
        return _clamp(1.0 - angle / math.pi)
    except Exception:
        return 0.0


def _extract_first_two_points(text: str) -> Optional[List[List[int]]]:
    pts: List[List[int]] = []
    for m in POINT_TAG_COORD_RE.finditer(text or ""):
        pts.append([int(m.group(1)), int(m.group(2))])
        if len(pts) == 2:
            return pts
    return None


def _kpt_score(gt_kps: Any, gt_hand_bbox: Any, pred_ray: Optional[Dict[str, List[int]]], response: str) -> float:
    pred_kps = _extract_first_two_points(response)
    if pred_kps is None and pred_ray is not None:
        pred_kps = [pred_ray["start"], pred_ray["end"]]
    if not (isinstance(gt_kps, list) and len(gt_kps) == 2 and isinstance(pred_kps, list) and len(pred_kps) == 2):
        return 0.0
    try:
        d0 = math.hypot(float(pred_kps[0][0]) - float(gt_kps[0][0]), float(pred_kps[0][1]) - float(gt_kps[0][1]))
        d1 = math.hypot(float(pred_kps[1][0]) - float(gt_kps[1][0]), float(pred_kps[1][1]) - float(gt_kps[1][1]))
        mpjpe = (d0 + d1) / 2.0
        scale = 1.0
        if isinstance(gt_hand_bbox, list) and len(gt_hand_bbox) == 4:
            gx1, gy1, gx2, gy2 = map(float, gt_hand_bbox)
            diag = math.hypot(abs(gx2 - gx1), abs(gy2 - gy1))
            if diag > 1e-8:
                scale = diag
        return _clamp(1.0 - mpjpe / scale)
    except Exception:
        return 0.0


def _count_draw_ray_tools(response: str) -> int:
    cnt = 0
    for m in TOOL_CALL_RE.finditer(response or ""):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        if _parse_draw_ray(obj) is not None:
            cnt += 1
    return cnt


def _count_obj_bboxes_after_last_tool_call(response: str) -> int:
    text = response or ""
    matches = list(TOOL_CALL_RE.finditer(text))
    tail = text[matches[-1].end():] if matches else text
    return len(_extract_tagged_boxes(tail))


def _extract_obj_pred_box_after_last_tool_call(response: str) -> Optional[List[int]]:
    text = response or ""
    matches = list(TOOL_CALL_RE.finditer(text))
    tail = text[matches[-1].end():] if matches else text
    boxes = _extract_tagged_boxes(tail)
    return boxes[-1] if boxes else None


def _stage2_format(response: str) -> float:
    obj_box_ok = 1.0 if _count_obj_bboxes_after_last_tool_call(response) >= 1 else 0.0
    obj_ref_ok = 1.0 if CAPTION_RE.search(response or "") else 0.0
    return _clamp(obj_box_ok * 0.9 + obj_ref_ok * 0.1)


def _neg_score_0_10(response: str) -> float:
    has_tool_call = _count_draw_ray_tools(response) > 0
    has_bbox = len(BOX_TAG_RE.findall(response or "")) > 0
    has_point = len(POINT_TAG_RE.findall(response or "")) > 0
    has_caption = len(CAPTION_RE.findall(response or "")) > 0
    penalty = 2.5 * float(has_tool_call + has_bbox + has_point + has_caption)
    return _clamp(10.0 - penalty)


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for ri in reward_inputs:
        response = ri.get("response", "") or ""
        gt_raw = ri.get("ground_truth")

        # Negative sample logic kept, only rescaled to [0,10].
        if _gt_is_none_like(gt_raw):
            neg = _neg_score_0_10(response)
            out.append({"overall": neg, "reward": neg, "neg": neg})
            continue

        gt = _safe_json_load(gt_raw)
        gt_hand_bbox = gt.get("hand_bbox")
        gt_ray = gt.get("pointing_ray")
        gt_kps = gt.get("pointing_keypoints")
        if gt_kps is None and isinstance(gt_ray, dict) and "start" in gt_ray and "end" in gt_ray:
            gt_kps = [gt_ray["start"], gt_ray["end"]]
        gt_obj_bbox = gt.get("obj_bbox")

        pred_hand_bbox = _pick_pred_box(response, first=True)
        pred_obj_bbox = _extract_obj_pred_box_after_last_tool_call(response)
        pred_ray = _extract_ray(response)

        hand_iou = _clamp(_iou(pred_hand_bbox, gt_hand_bbox))
        ray_cos = _ray_cos(gt_ray, pred_ray)
        kpt_score = _kpt_score(gt_kps, gt_hand_bbox, pred_ray, response)
        obj_iou = _clamp(_iou(pred_obj_bbox, gt_obj_bbox))
        stage2_format = _stage2_format(response)

        base = hand_iou + ray_cos + kpt_score + obj_iou * 5.0 + stage2_format * 2.0
        base = _clamp(base, 0.0, 10.0)

        # No gate: only penalize repeated tool calls / repeated object bboxes.
        tool_cnt = _count_draw_ray_tools(response)
        obj_box_cnt = _count_obj_bboxes_after_last_tool_call(response)
        tool_penalty = 1.0 / float(max(1, tool_cnt))
        if obj_box_cnt > 0:
            bbox_penalty = 1.0 / float(obj_box_cnt)
        else:
            bbox_penalty = 0.1
        reward = _clamp(base * tool_penalty * bbox_penalty, 0.0, 10.0)

        out.append(
            {
                "overall": reward,
                "reward": reward,
                "hand_iou": hand_iou,
                "ray_cos": ray_cos,
                "kpt_score": kpt_score,
                "obj_iou": obj_iou,
                "format": stage2_format,
                "stage2_format": stage2_format,
                "tool_count": float(tool_cnt),
                "obj_box_count": float(obj_box_cnt),
                "tool_penalty": tool_penalty,
                "bbox_penalty": bbox_penalty,
                "base_score": base,
            }
        )
    return out
