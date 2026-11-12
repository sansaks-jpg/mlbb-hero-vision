"""Build Google Colab notebook for training MLBB Hero Vision via Knowledge Distillation."""

from __future__ import annotations

import json
from pathlib import Path

OUT_NB = Path(__file__).resolve().parent / "mlbb_hero_distill_colab.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MLBB Hero Vision — Knowledge Distillation (Colab T4 GPU)\n",
            "Trains a **ConvNeXt-Tiny** student model distilled from **Gemini Embedding 2** vectors with tournament overlay simulation (Blue/Red team gradients, battle spells, IGN overlays, circular masks, JPEG artifacts) and exports a standalone `hero_student.onnx`.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q timm onnx onnxscript onnxruntime huggingface_hub\n",
            "!git clone -q https://github.com/sansaks-jpg/mlbb-hero-vision.git\n",
            "%cd mlbb-hero-vision\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Option A: Mount Google Drive if your hero images live in /content/drive/MyDrive/hero-ml\n",
            "# from google.colab import drive\n",
            "# drive.mount('/content/drive')\n",
            "\n",
            "# Point --data-dir to your dataset folder (either <data-dir>/<HeroName>/*.png or multi-source layout)\n",
            "# Optionally include _distill/gemini_embedding_index.npz & gemini_embedding_meta.json for teacher distillation.\n",
            "!python train/train_distill.py --data-dir /content/drive/MyDrive/hero-ml --out-dir ./models --epochs 18 --batch-size 64\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test the newly exported ONNX model offline\n",
            "from mlbb_hero_vision import HeroVision\n",
            "vision = HeroVision(model_dir='./models', auto_download=False)\n",
            "print('Loaded classes:', len(vision.labels))\n",
        ],
    },
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
OUT_NB.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Wrote notebook: {OUT_NB}")
