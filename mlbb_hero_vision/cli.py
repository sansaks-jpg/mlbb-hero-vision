"""CLI entrypoint for 100% offline MLBB Hero Vision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .student import HeroVision, ensure_model_files

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline MLBB Hero Vision (ConvNeXt-Tiny ONNX)")
    parser.add_argument("--image", type=str, help="Path to a single cropped hero slot image")
    parser.add_argument("--folder", type=str, help="Directory of cropped hero images to evaluate in batch")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates/probabilities")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing hero_student.onnx & index files")
    parser.add_argument("--download-only", action="store_true", help="Download model weights from Hugging Face and exit")
    args = parser.parse_args()

    if args.download_only:
        out = ensure_model_files(args.model_dir, verbose=True)
        print(f"Model files ready at: {out}")
        return

    vision = HeroVision(model_dir=args.model_dir, auto_download=True)

    if args.image:
        result = vision.predict(args.image, top_k=args.top_k)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.folder:
        folder = Path(args.folder)
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        outputs = []
        for p in files:
            res = vision.predict(p, top_k=args.top_k)
            outputs.append({"file": p.name, "hero": res["hero"], "confidence": res["confidence"], "latency_ms": res["latency_ms"]})
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
