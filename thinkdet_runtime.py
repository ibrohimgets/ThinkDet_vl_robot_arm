"""Runtime inference adapter for ThinkDet and GroundingDINO."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


DEFAULT_GD_CONFIG_REL = (
    "GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DEFAULT_GD_WEIGHTS_REL = "GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
DEFAULT_INTERNVL_REL = "InternVL3_5-1B"
DEFAULT_THINKDET_CKPT_REL = (
    "thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/thinkdet_unified_epoch5.pth"
)


def _normalize_query(text: str) -> str:
    text = " ".join(str(text).strip().split())
    if not text:
        raise ValueError("query must be non-empty")
    return text if text.endswith(".") else f"{text} ."


def _build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image_pil: Image.Image):
        w, h = image_pil.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = image_pil.resize((nw, nh), Image.BILINEAR)
        return normalize(TF.to_tensor(img))

    return transform


def _build_internvl_transform():
    return T.Compose(
        [
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _get_special_tokens(model):
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens
    tokens = getattr(model, "special_tokens", None)
    if tokens is not None:
        return tokens
    return model.tokenizer.all_special_ids


def _build_positive_map_for_query(tokenizer, special_tokens, query_text, max_text_len=512):
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    positive_map = torch.zeros(1, max_text_len, dtype=torch.float32)
    special_set = {int(x) for x in special_tokens}
    for pos, tok_id in enumerate(input_ids.tolist()):
        if pos >= max_text_len:
            break
        if tok_id not in special_set:
            positive_map[0, pos] = 1.0
    row_sum = positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return positive_map / row_sum


def _score_outputs(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


class ThinkDetInference:
    """Inference wrapper exposing a simple `predict(image, query)` API."""

    def __init__(
        self,
        repo_root: str = "",
        backend: str = "auto",
        device: str = "auto",
        gd_config: str = "",
        gd_weights: str = "",
        internvl_path: str = "",
        thinkdet_checkpoint: str = "",
        extract_layer: Optional[int] = None,
        extract_layers: Optional[Sequence[int]] = None,
        layer_fusion: Optional[str] = None,
    ):
        self.repo_root = self._resolve_repo_root(repo_root)
        self._bootstrap_repo_imports(self.repo_root)

        from groundingdino.util.inference import load_model as load_gd_model
        from groundingdino.util.misc import NestedTensor
        from thinkdet.models.arch import DEFAULT_INJECTION_LAYERS, ThinkDetModel

        self._load_gd_model = load_gd_model
        self._NestedTensor = NestedTensor
        self._ThinkDetModel = ThinkDetModel
        self._default_injection_layers = DEFAULT_INJECTION_LAYERS

        self.device = self._resolve_device(device)
        self.device_str = str(self.device)

        self.gd_config = self._resolve_path(gd_config, DEFAULT_GD_CONFIG_REL)
        self.gd_weights = self._resolve_path(gd_weights, DEFAULT_GD_WEIGHTS_REL)
        self.internvl_path = self._resolve_path(internvl_path, DEFAULT_INTERNVL_REL)
        self.thinkdet_checkpoint = self._resolve_path(
            thinkdet_checkpoint,
            DEFAULT_THINKDET_CKPT_REL,
        )

        self.extract_layer = extract_layer
        self.extract_layers = list(extract_layers) if extract_layers else None
        self.layer_fusion = layer_fusion
        self.backend = self._resolve_backend(backend)

        self.dino_transform = _build_dino_transform()
        self.internvl_transform = _build_internvl_transform()

        self.model = None
        self.model_meta: Dict[str, Any] = {}
        self._load_runtime()

    def _resolve_repo_root(self, explicit_repo_root: str) -> Path:
        candidates: List[Path] = []
        if explicit_repo_root:
            candidates.append(Path(explicit_repo_root))

        env_root = os.environ.get("THINKDET_REPO_ROOT", "")
        if env_root:
            candidates.append(Path(env_root))

        here = Path(__file__).resolve()
        candidates.extend([Path.cwd(), here.parent, here.parent.parent])
        for base in list(candidates):
            candidates.extend(base.parents)

        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "thinkdet").exists() and (candidate / "GroundingDINO").exists():
                return candidate
        raise FileNotFoundError(
            "Could not locate the repo root. Set THINKDET_REPO_ROOT or pass repo_root."
        )

    def _bootstrap_repo_imports(self, repo_root: Path):
        root_str = str(repo_root)
        gd_pkg = str(repo_root / "GroundingDINO" / "GroundingDINO")
        for entry in (root_str, gd_pkg):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    def _resolve_device(self, device: str) -> torch.device:
        device = str(device or "auto").strip().lower()
        if device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_path(self, explicit_path: str, default_rel: str) -> str:
        explicit_path = str(explicit_path or "").strip()
        path = Path(explicit_path) if explicit_path else (self.repo_root / default_rel)
        return str(path.resolve())

    def _resolve_backend(self, backend: str) -> str:
        backend = str(backend or "auto").strip().lower()
        if backend in {"gdino", "grounding_dino", "baseline"}:
            backend = "groundingdino"
        if backend not in {"auto", "thinkdet", "groundingdino"}:
            raise ValueError(f"Unsupported backend '{backend}'")

        thinkdet_ready = all(
            Path(path).exists()
            for path in (self.gd_config, self.gd_weights, self.internvl_path, self.thinkdet_checkpoint)
        )
        baseline_ready = all(Path(path).exists() for path in (self.gd_config, self.gd_weights))

        if backend == "auto":
            if thinkdet_ready:
                return "thinkdet"
            if baseline_ready:
                return "groundingdino"
            raise FileNotFoundError(
                "Auto backend could not find a valid ThinkDet or GroundingDINO runtime."
            )

        if backend == "thinkdet" and not thinkdet_ready:
            raise FileNotFoundError(
                "ThinkDet backend requested but one or more required paths are missing."
            )
        if backend == "groundingdino" and not baseline_ready:
            raise FileNotFoundError(
                "GroundingDINO backend requested but config or weights are missing."
            )
        return backend

    def _load_runtime(self):
        if self.backend == "groundingdino":
            model = self._load_gd_model(self.gd_config, self.gd_weights, device="cpu")
            self.model = model.to(self.device).eval()
            self.model_meta = {
                "backend": self.backend,
                "device": self.device_str,
                "gd_config": self.gd_config,
                "gd_weights": self.gd_weights,
            }
            return

        checkpoint = torch.load(self.thinkdet_checkpoint, map_location="cpu")
        checkpoint_extract_layers = checkpoint.get("extract_layers") or [
            checkpoint.get("extract_layer", 9)
        ]
        ext_layers = self.extract_layers or (
            [int(self.extract_layer)] if self.extract_layer is not None else checkpoint_extract_layers
        )
        ext_layers = sorted({int(x) for x in ext_layers})
        layer_fusion = self.layer_fusion or checkpoint.get("layer_fusion", "mean") or "mean"
        injection_layers = checkpoint.get(
            "injection_layers",
            self._default_injection_layers,
        )
        tma_m = checkpoint.get("tma_m", 8)
        tma_n_heads = checkpoint.get("tma_n_heads", 8) or 8
        fusion_mode = checkpoint.get("fusion_mode", "concat") or "concat"

        grounding_dino = self._load_gd_model(self.gd_config, self.gd_weights, device="cpu")
        model = self._ThinkDetModel(
            grounding_dino=grounding_dino,
            internvl_path=self.internvl_path,
            extract_layer=max(ext_layers),
            extract_layers=ext_layers,
            layer_fusion=layer_fusion,
            injection_layers=injection_layers,
            tma_m=tma_m,
            tma_n_heads=tma_n_heads,
            fusion_mode=fusion_mode,
            preserve_kd_enabled=False,
        )
        state_dict = checkpoint.get("trainable_state_dict") or checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError(
                f"Checkpoint {self.thinkdet_checkpoint} does not contain a supported state dict."
            )
        incompat = model.load_state_dict(state_dict, strict=False)
        self.model = model.to(self.device).eval()
        self.model_meta = {
            "backend": self.backend,
            "device": self.device_str,
            "gd_config": self.gd_config,
            "gd_weights": self.gd_weights,
            "internvl_path": self.internvl_path,
            "thinkdet_checkpoint": self.thinkdet_checkpoint,
            "extract_layers": ext_layers,
            "layer_fusion": layer_fusion,
            "injection_layers": injection_layers,
            "tma_m": tma_m,
            "tma_n_heads": tma_n_heads,
            "fusion_mode": fusion_mode,
            "missing_keys": list(getattr(incompat, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompat, "unexpected_keys", [])),
        }

    def describe(self) -> str:
        pieces = [f"backend={self.backend}", f"device={self.device_str}"]
        if self.backend == "thinkdet":
            pieces.append(f"ckpt={Path(self.thinkdet_checkpoint).name}")
        else:
            pieces.append(f"weights={Path(self.gd_weights).name}")
        return " | ".join(pieces)

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a PIL.Image or numpy.ndarray")

        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if image.shape[2] == 3:
            rgb = image[:, :, ::-1]
            return Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
        if image.shape[2] == 4:
            rgba = image[:, :, [2, 1, 0, 3]]
            return Image.fromarray(np.ascontiguousarray(rgba)).convert("RGB")
        raise ValueError(f"Unsupported image channel count: {image.shape[2]}")

    def _make_nested_tensor(self, image_pil: Image.Image):
        dino_t = self.dino_transform(image_pil).unsqueeze(0).to(self.device)
        mask = torch.zeros(
            1,
            dino_t.shape[2],
            dino_t.shape[3],
            dtype=torch.bool,
            device=self.device,
        )
        return self._NestedTensor(dino_t, mask)

    def _run_groundingdino(self, image_pil: Image.Image, query_text: str):
        nested = self._make_nested_tensor(image_pil)
        return self.model(samples=nested, captions=[query_text]), {}

    def _run_thinkdet(self, image_pil: Image.Image, raw_query: str, query_text: str):
        nested = self._make_nested_tensor(image_pil)
        ivl_t = self.internvl_transform(image_pil).unsqueeze(0).to(self.device)
        outputs, aux = self.model(
            ivl_t,
            [raw_query],
            {"samples": nested, "captions": [query_text]},
        )
        return outputs, aux

    def _outputs_to_results(
        self,
        outputs,
        image_size,
        query_text: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        img_w, img_h = image_size
        tokenizer_model = self.model.grounding_dino if self.backend == "thinkdet" else self.model
        positive_map_norm = _build_positive_map_for_query(
            tokenizer_model.tokenizer,
            _get_special_tokens(tokenizer_model),
            query_text,
            max_text_len=512,
        )
        scores, boxes = _score_outputs(outputs, positive_map_norm)
        if scores.numel() == 0:
            return []

        vals, idx = scores.topk(min(int(top_k), int(scores.shape[0])))
        label = query_text.rstrip(" .")
        results: List[Dict[str, Any]] = []
        for rank, det_idx in enumerate(idx.tolist()):
            score = float(vals[rank].item())
            if score < float(score_threshold):
                continue
            cx, cy, bw, bh = boxes[det_idx].detach().cpu().tolist()
            x1 = max(0.0, min(float(img_w), (cx - bw / 2.0) * img_w))
            y1 = max(0.0, min(float(img_h), (cy - bh / 2.0) * img_h))
            x2 = max(0.0, min(float(img_w), (cx + bw / 2.0) * img_w))
            y2 = max(0.0, min(float(img_h), (cy + bh / 2.0) * img_h))
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            results.append(
                {
                    "bbox": bbox,
                    "box_abs_xyxy": list(bbox),
                    "score": score,
                    "label": label,
                    "backend": self.backend,
                }
            )
        return results

    @torch.no_grad()
    def predict(
        self,
        image: Any,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        image_pil = self._to_pil(image)
        raw_query = " ".join(str(query).strip().split())
        query_text = _normalize_query(raw_query)

        if self.backend == "thinkdet":
            outputs, _ = self._run_thinkdet(image_pil, raw_query, query_text)
        else:
            outputs, _ = self._run_groundingdino(image_pil, query_text)

        return self._outputs_to_results(
            outputs,
            image_pil.size,
            query_text,
            top_k=top_k,
            score_threshold=score_threshold,
        )
