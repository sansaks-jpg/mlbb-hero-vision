"""100% offline ONNX Student matcher distilled from Gemini Embedding 2."""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _resolve_default_model_dir() -> Path:
    env_dir = os.environ.get("MLBB_HERO_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    repo_models = PACKAGE_ROOT / "models"
    if repo_models.is_dir() and os.access(repo_models, os.W_OK):
        return repo_models
    return Path.home() / ".cache" / "mlbb-hero-vision"


DEFAULT_MODEL_DIR = _resolve_default_model_dir()
HF_REPO_ID = "sansaks/mlbb-hero-vision"
GH_RELEASE_TAG = "v1.1.0"
GH_REPO_SLUG = "sansaks-jpg/mlbb-hero-vision"

MODEL_FILES = (
    "hero_student.onnx",
    "hero_student_db.npz",
    "hero_student_labels.json",
    "hero_student_meta.json",
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def is_student_ready(model_dir: str | Path | None = None) -> bool:
    """Return True when all four offline student artifacts exist in model_dir."""
    root = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    return all((root / fname).is_file() for fname in MODEL_FILES)


def ensure_model_files(model_dir: str | Path | None = None, verbose: bool = True) -> Path:
    """Download missing ONNX model and vector index artifacts from Hugging Face / GitHub Releases."""
    root = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    root.mkdir(parents=True, exist_ok=True)
    for fname in MODEL_FILES:
        target = root / fname
        if target.is_file() and target.stat().st_size > 0:
            continue
        urls = [
            f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{fname}",
            f"https://github.com/{GH_REPO_SLUG}/releases/download/{GH_RELEASE_TAG}/{fname}",
        ]
        last_err: Exception | None = None
        for url in urls:
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                if verbose:
                    print(f"[mlbb-hero-vision] Downloading {fname} from {url} ...")
                req = urllib.request.Request(url, headers={"User-Agent": "mlbb-hero-vision/1.0"})
                with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as fh:
                    shutil.copyfileobj(resp, fh, length=1024 * 1024)
                tmp.replace(target)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if tmp.exists():
                    tmp.unlink()
        if last_err is not None:
            raise RuntimeError(f"Failed to download {fname}: {last_err}") from last_err
    return root


def _preprocess(img: Image.Image) -> np.ndarray:
    """Composite transparent inputs onto neutral dark blue and normalize to ImageNet stats."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (14, 19, 32, 255))
        bg.alpha_composite(rgba)
        rgb = bg.convert("RGB")
    else:
        rgb = img.convert("RGB")
    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - MEAN) / STD
    return arr[None, :, :, :].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class HeroVision:
    """Offline MLBB hero recognizer powered by distilled ConvNeXt-Tiny ONNX."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        hero_assets_dir: str | Path | None = None,
        auto_download: bool = True,
        providers: list[str] | None = None,
        min_confidence: float = 0.52,
        min_margin: float = 0.30,
        min_proto_cos: float = 0.27,
        min_db_sim: float = 0.885,
        max_entropy: float = 2.75,
    ) -> None:
        import onnxruntime as ort

        root = ensure_model_files(model_dir) if auto_download else Path(model_dir or DEFAULT_MODEL_DIR)
        if not is_student_ready(root):
            raise FileNotFoundError(f"Missing model artifacts in {root}. Call ensure_model_files().")

        self.model_dir = root
        self.hero_assets_dir = Path(hero_assets_dir) if hero_assets_dir else None
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.min_proto_cos = min_proto_cos
        self.min_db_sim = min_db_sim
        self.max_entropy = max_entropy
        self.sess = ort.InferenceSession(
            str(root / "hero_student.onnx"),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.db_mat = np.load(root / "hero_student_db.npz")["vectors"].astype(np.float32)
        self.meta: list[dict[str, str]] = json.loads((root / "hero_student_meta.json").read_text(encoding="utf-8"))
        self.labels: list[str] = json.loads((root / "hero_student_labels.json").read_text(encoding="utf-8"))

    def predict(self, image: Image.Image | str | Path, top_k: int = 5) -> dict[str, Any]:
        """Classify a cropped MLBB hero box (draft pick, ban slot, or in-game HUD) offline."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                return self._predict_pil(im, top_k=top_k)
        return self._predict_pil(image, top_k=top_k)

    def _predict_pil(self, img: Image.Image, top_k: int = 5) -> dict[str, Any]:
        t0 = time.perf_counter()
        x = _preprocess(img)
        emb, logits = self.sess.run(None, {"pixel_values": x})
        qv = emb[0].astype(np.float32)
        raw_logits = logits[0].astype(np.float32)
        probs = _softmax(raw_logits)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        sorted_probs = np.sort(probs)
        top_cls_idx = int(np.argmax(probs))
        top_cls_name = self.labels[top_cls_idx]
        top_cls_prob = float(probs[top_cls_idx])
        prob_margin = float(top_cls_prob - sorted_probs[-2]) if len(sorted_probs) > 1 else top_cls_prob
        proto_cos = float(raw_logits[top_cls_idx] / 18.0)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))

        scores = self.db_mat @ qv
        order = np.argsort(-scores)
        db_top1 = float(scores[order[0]]) if len(order) > 0 else 0.0
        by_hero: dict[str, dict[str, Any]] = {}
        for idx in order:
            m = self.meta[int(idx)]
            hero = m["hero"]
            if hero not in by_hero:
                entry: dict[str, Any] = {
                    "hero": hero,
                    "score": round(float(scores[idx]), 4),
                    "matched_shape": m.get("shape", ""),
                    "matched_source": m.get("source", ""),
                    "asset_relpath": m.get("path", ""),
                }
                if self.hero_assets_dir is not None:
                    entry["matched_path"] = str((self.hero_assets_dir / m["path"]).resolve())
                by_hero[hero] = entry
                if len(by_hero) >= top_k:
                    break

        rejection_reasons: list[str] = []
        if top_cls_name == "EMPTY_OR_UNKNOWN":
            rejection_reasons.append("empty_or_unknown_class")
        else:
            if top_cls_prob < self.min_confidence:
                rejection_reasons.append(f"low_confidence({top_cls_prob:.3f}<{self.min_confidence})")
            if prob_margin < self.min_margin:
                rejection_reasons.append(f"low_margin({prob_margin:.3f}<{self.min_margin})")
            if proto_cos < self.min_proto_cos:
                rejection_reasons.append(f"low_proto_cos({proto_cos:.3f}<{self.min_proto_cos})")
            if db_top1 < self.min_db_sim:
                rejection_reasons.append(f"low_db_sim({db_top1:.3f}<{self.min_db_sim})")
            if entropy > self.max_entropy:
                rejection_reasons.append(f"high_entropy({entropy:.2f}>{self.max_entropy})")

        accepted = len(rejection_reasons) == 0
        return {
            "hero": top_cls_name if accepted else None,
            "is_mlbb_hero": accepted,
            "raw_choice": top_cls_name,
            "confidence": round(top_cls_prob, 4),
            "prob_margin": round(prob_margin, 4),
            "proto_cos": round(proto_cos, 4),
            "db_top1": round(db_top1, 4),
            "entropy": round(entropy, 4),
            "rejection_reason": None if accepted else ", ".join(rejection_reasons),
            "latency_ms": latency_ms,
            "engine": "distilled-convnext-tiny-onnx",
            "top_probabilities": [
                {"hero": self.labels[int(i)], "prob": round(float(probs[int(i)]), 4)}
                for i in np.argsort(-probs)[:top_k]
            ],
            "top_candidates": list(by_hero.values()),
        }
