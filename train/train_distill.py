"""Portable Knowledge Distillation + Tournament Overlay Simulation trainer.

Trains a ConvNeXt-Tiny ONNX student from Gemini Embedding 2 vectors (or supervised
alone if teacher embeddings are omitted) with heavy MLBB tournament augmentations:
- Partial face/upper-body/aspect-ratio crops + horizontal flips
- Circular draft/ban slot masking
- Blue Team, Red Team, and Neutral Dark edge gradients (left / right / bottom)
- Real battle spell icon overlays + national flag badges + player IGN bars
- Brightness/contrast jitter, Gaussian blur, JPEG compression artifacts
- Explicit EMPTY_OR_UNKNOWN negative class for unpicked/locked slots
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALIASES = {
    "arlot": "Arlott",
    "beleric": "Belerick",
    "carmila": "Carmilla",
    "change": "Chang'e",
    "lapulapu": "Lapu-Lapu",
    "luoyi": "Luo Yi",
    "popolandkupa": "Popol and Kupa",
    "xborg": "X.Borg",
    "yisunshin": "Yi Sun-shin",
    "yuzhong": "Yu Zhong",
    "zhuxin": "Zhu Xin",
    "minsithar": "Minsitthar",
}

IGN_SAMPLES = [
    "OP Yuna", "OP Arr'De'Kyou~", "RRQ Skylar", "ONIC Kairi", "EVOS Tazz",
    "TLID Aran", "BTR Kenn", "GEEK Baloyskie", "AE Nino", "Player 1", "Sanz.",
    "Kiboy!", "CW", "Super Marco", "KarlTzy", "Kelra", "FlapTzy",
]

NORM_TF = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

SHAPE_TRAIN_TF = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.45, hue=0.12)], p=0.65),
        transforms.RandomGrayscale(p=0.20),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.18, p=1.0)], p=0.30),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4))], p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.30, scale=(0.04, 0.16), ratio=(0.4, 2.5), value=0),
    ]
)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def composite_rgba(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (14, 19, 32, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB")


def draw_random_flag(size: int) -> Image.Image:
    flag = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    palettes = [
        [(200, 30, 40, 255), (245, 245, 245, 255)],
        [(16, 120, 60, 255), (245, 245, 245, 255)],
        [(20, 50, 150, 255), (235, 200, 30, 255), (200, 30, 40, 255)],
        [(15, 20, 40, 255), (220, 40, 40, 255), (245, 245, 245, 255)],
    ]
    cols = random.choice(palettes)
    stripe_h = max(2, size // len(cols))
    temp = Image.new("RGBA", (size, size))
    td = ImageDraw.Draw(temp)
    for i, c in enumerate(cols):
        td.rectangle((0, i * stripe_h, size, (i + 1) * stripe_h + 2), fill=c)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    flag.paste(temp, (0, 0), mask)
    ImageDraw.Draw(flag).ellipse((0, 0, size - 1, size - 1), outline=(220, 230, 245, 200), width=1)
    return flag


def make_empty_slot(size: int = 224) -> Image.Image:
    base = (random.randint(10, 32), random.randint(14, 42), random.randint(24, 60))
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)
    if random.random() < 0.7:
        draw.rectangle((8, 8, size - 9, size - 9), outline=(70, 95, 140), width=random.randint(2, 4))
    if random.random() < 0.5:
        draw.text((size // 3, size // 2 - 6), random.choice(["BAN", "PICK", "...", "LOCKED", ""]), fill=(140, 160, 190))
    return img


def simulate_tournament_box(img: Image.Image, spell_imgs: list[Image.Image], target_size: int = 224) -> Image.Image:
    rgb = composite_rgba(img)
    w, h = rgb.size
    mode = random.random()
    if mode < 0.25:
        scale = random.uniform(0.82, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0 = random.randint(0, max(0, w - cw))
        y0 = random.randint(0, max(0, h - ch))
        rgb = rgb.crop((x0, y0, x0 + cw, y0 + ch))
    elif mode < 0.65:
        cw = int(w * random.uniform(0.7, 1.0))
        ch = min(h, int(cw * random.uniform(0.55, 0.85)))
        x0 = random.randint(0, max(0, w - cw))
        y0 = random.randint(0, max(0, int((h - ch) * 0.45)))
        rgb = rgb.crop((x0, y0, x0 + cw, y0 + ch))
    else:
        side = int(min(w, h) * random.uniform(0.55, 0.9))
        x0 = random.randint(0, max(0, w - side))
        y0 = random.randint(0, max(0, int((h - side) * 0.4)))
        rgb = rgb.crop((x0, y0, x0 + side, y0 + side))

    if random.random() < 0.5:
        rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    canvas = Image.new("RGBA", (target_size, target_size), (14, 22, 42, 255))
    if random.random() < 0.5:
        canvas.paste(rgb.resize((target_size, target_size), Image.Resampling.BILINEAR).convert("RGBA"), (0, 0))
    else:
        rw, rh = rgb.size
        nh = max(32, min(target_size, int(target_size / (rw / max(1, rh)))))
        canvas.paste(rgb.resize((target_size, nh), Image.Resampling.BILINEAR).convert("RGBA"), (0, (target_size - nh) // 2))

    if random.random() < 0.30:
        mask = Image.new("L", (target_size, target_size), 0)
        pad = random.randint(4, 18)
        ImageDraw.Draw(mask).ellipse((pad, pad, target_size - pad, target_size - pad), fill=255)
        bg = Image.new("RGBA", (target_size, target_size), (10, 14, 24, 255))
        bg.paste(canvas, (0, 0), mask)
        canvas = bg

    # Blue Team, Red Team, or Neutral Dark edge gradients
    if random.random() < 0.70:
        grad = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grad)
        team_col = random.choice([
            (14, 52, 132),   # Blue team
            (20, 82, 168),   # Bright blue team
            (138, 22, 34),   # Red team
            (172, 30, 44),   # Bright red team
            (12, 18, 34),    # Neutral dark
        ])
        side = random.choice(["left", "right", "bottom"])
        if side == "right":
            for x in range(target_size // 3, target_size, 4):
                alpha = int(205 * ((x - target_size // 3) / (target_size * 2 / 3)))
                gd.rectangle((x, 0, x + 4, target_size), fill=(*team_col, alpha))
        elif side == "left":
            for x in range(0, (target_size * 2) // 3, 4):
                alpha = int(205 * (1.0 - x / (target_size * 2 / 3)))
                gd.rectangle((x, 0, x + 4, target_size), fill=(*team_col, alpha))
        else:
            for y in range(target_size // 2, target_size, 4):
                alpha = int(205 * ((y - target_size // 2) / (target_size / 2)))
                gd.rectangle((0, y, target_size, y + 4), fill=(*team_col, alpha))
        canvas.alpha_composite(grad)

    if spell_imgs and random.random() < 0.50:
        sp = random.choice(spell_imgs)
        sp_sz = random.randint(28, 48)
        sp_res = sp.resize((sp_sz, sp_sz), Image.Resampling.BILINEAR)
        px = random.randint(target_size - sp_sz - 36, target_size - sp_sz - 6)
        py = random.randint(10, 38)
        canvas.alpha_composite(sp_res, (px, py))

    if random.random() < 0.45:
        fl_sz = random.randint(22, 34)
        canvas.alpha_composite(draw_random_flag(fl_sz), (random.randint(8, 22), random.randint(14, 34)))

    if random.random() < 0.55:
        draw = ImageDraw.Draw(canvas)
        if random.random() < 0.6:
            draw.rectangle((0, target_size - 28, target_size, target_size), fill=(10, 16, 30, 190))
            draw.text((12, target_size - 20), random.choice(IGN_SAMPLES), fill=(225, 232, 245, 240))
        if random.random() < 0.5:
            draw.rectangle((0, 0, target_size, 22), fill=(10, 16, 30, 190))
            draw.text((12, 5), random.choice(IGN_SAMPLES), fill=(225, 232, 245, 230))

    out = canvas.convert("RGB")
    if random.random() < 0.6:
        out = ImageEnhance.Brightness(out).enhance(random.uniform(0.80, 1.20))
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.1)))
    if random.random() < 0.5:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=random.randint(45, 90))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


class TournamentDistillDataset(Dataset):
    def __init__(self, recs, h2i, spell_imgs, samples_per_epoch=9600, empty_ratio=0.05, is_train=True):
        self.records = recs
        self.h2i = h2i
        self.spell_imgs = spell_imgs
        self.samples_per_epoch = samples_per_epoch
        self.empty_ratio = empty_ratio
        self.is_train = is_train
        self.zero_vec = np.zeros(768, dtype=np.float32)

    def __len__(self):
        return self.samples_per_epoch if self.is_train else len(self.records)

    def __getitem__(self, idx):
        if self.is_train and random.random() < self.empty_ratio:
            return NORM_TF(make_empty_slot(224)), self.h2i["EMPTY_OR_UNKNOWN"], torch.from_numpy(self.zero_vec), torch.tensor(0.0)
        rec = self.records[random.randrange(len(self.records)) if self.is_train else idx]
        with Image.open(rec["path"]) as im:
            proc = simulate_tournament_box(im, self.spell_imgs, 224) if self.is_train else composite_rgba(im).resize((224, 224), Image.Resampling.BILINEAR)
        tvec = rec["teacher_vec"]
        has_tv = 1.0 if tvec is not None else 0.0
        tv_arr = torch.from_numpy(tvec.copy()) if tvec is not None else torch.from_numpy(self.zero_vec)
        tf = SHAPE_TRAIN_TF if (self.is_train and random.random() < 0.50) else NORM_TF
        return tf(proc), self.h2i[rec["hero"]], tv_arr, torch.tensor(has_tv)


class HeroStudentNet(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, emb_dim: int = 768):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.LayerNorm(emb_dim))
        self.prototypes = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.02)
        self.logit_scale = 18.0

    def forward(self, x):
        feat = self.backbone(x)
        emb = F.normalize(self.proj(feat), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        logits = self.logit_scale * (emb @ proto.t())
        return emb, logits


def resolve_hero_dir(data_dir: str | None) -> Path:
    if data_dir:
        p = Path(data_dir)
        if p.is_file():
            extract = Path("/tmp/extract_hero")
            with tarfile.open(p, "r") as tf:
                tf.extractall(extract)
            return extract / "hero-ml"
        return p
    found = list(Path("/kaggle/input").rglob("hero_ml_bundle.bin"))
    if found:
        extract = Path("/tmp/extract_hero")
        if not (extract / "hero-ml").exists():
            with tarfile.open(found[0], "r") as tf:
                tf.extractall(extract)
        return extract / "hero-ml"
    raise FileNotFoundError("Pass --data-dir pointing to hero-ml image directory or tar bundle.")


def train(args: argparse.Namespace) -> None:
    hero_dir = resolve_hero_dir(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_meta_p = hero_dir / "_distill" / "gemini_embedding_meta.json"
    teacher_idx_p = hero_dir / "_distill" / "gemini_embedding_index.npz"
    rel_to_vec: dict[str, np.ndarray] = {}
    hero_proto: dict[str, np.ndarray] = {}
    canonical: dict[str, str] = {slugify(k): v for k, v in ALIASES.items()}

    if teacher_meta_p.exists() and teacher_idx_p.exists():
        teacher_meta = json.loads(teacher_meta_p.read_text(encoding="utf-8"))
        teacher_mat = np.load(teacher_idx_p)["vectors"].astype(np.float32)
        hero_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
        for m, vec in zip(teacher_meta, teacher_mat):
            canonical[slugify(m["hero"])] = m["hero"]
            p_norm = m["path"].replace("\\", "/")
            if "/hero-ml/" in p_norm:
                rel_to_vec[p_norm.split("/hero-ml/", 1)[1]] = vec
            hero_vecs[m["hero"]].append(vec)
        for h, vs in hero_vecs.items():
            mv = np.mean(vs, axis=0)
            hero_proto[h] = (mv / max(np.linalg.norm(mv), 1e-8)).astype(np.float32)

    records: list[dict] = []
    for p in sorted(hero_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(hero_dir)
        parts = rel.parts
        top = parts[0]
        if top.startswith("_"):
            continue
        stem = p.stem
        if top in {"asset_train", "mlbb-cdn-heroes"} and len(parts) >= 3:
            raw, shape = parts[1], (stem.split("_")[-1] if "_" in stem else top)
        elif top == "wiki-hero-icons":
            raw, shape = re.sub(r"^icon[-_]", "", stem, flags=re.I), "wiki-icon"
        elif top == "wiki-hero-portraits":
            raw, shape = re.sub(r"^portrait[-_]", "", stem, flags=re.I), "wiki-portrait"
        elif top == "heroban-canvas":
            raw, shape = stem, "ban"
        elif top in {"HeroPick1-canvas", "HeroPick-custom-draftpick"}:
            raw, shape = stem, "pick"
        elif len(parts) == 2:
            raw, shape = parts[0], "custom"
        else:
            raw, shape = stem, top
        hero = canonical.get(slugify(raw), re.sub(r"[_-]+", " ", raw).strip().title() or "Unknown")
        if hero_proto and hero not in hero_proto:
            continue
        rel_key = str(rel).replace("\\", "/")
        records.append(
            {
                "path": str(p),
                "rel": rel_key,
                "hero": hero,
                "shape": shape,
                "source": top,
                "teacher_vec": rel_to_vec.get(rel_key, hero_proto.get(hero)),
            }
        )

    heroes = sorted({r["hero"] for r in records}) + ["EMPTY_OR_UNKNOWN"]
    h2i = {h: i for i, h in enumerate(heroes)}
    spell_imgs = [Image.open(p).convert("RGBA") for p in sorted((hero_dir / "_distill" / "spells").glob("*.png"))] if (hero_dir / "_distill" / "spells").exists() else []
    print(f"Dataset: {len(records)} images across {len(heroes)-1} heroes | spells={len(spell_imgs)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeroStudentNet("convnext_tiny.fb_in22k_ft_in1k", len(heroes), emb_dim=768).to(device)
    loader = DataLoader(TournamentDistillDataset(records, h2i, spell_imgs, samples_per_epoch=args.samples_per_epoch), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = tot_cls = tot_dist = 0.0
        correct = total = 0
        for imgs, labels, tvecs, has_tvec in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            tvecs, has_tvec = tvecs.to(device), has_tvec.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                emb, logits = model(imgs)
                loss_cls = F.cross_entropy(logits, labels, label_smoothing=0.08)
                loss_dist = ((1.0 - (emb * tvecs).sum(dim=-1)) * has_tvec).sum() / has_tvec.sum().clamp(min=1.0)
                loss = loss_cls + 2.0 * loss_dist
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = imgs.size(0)
            tot_loss += loss.item() * bs
            tot_cls += loss_cls.item() * bs
            tot_dist += loss_dist.item() * bs
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += bs
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={tot_loss/total:.4f} (cls={tot_cls/total:.4f}, dist={tot_dist/total:.4f}) | acc={correct/total*100:.2f}%")

    tmp_onnx = out_dir / "_tmp_student.onnx"
    onnx_path = out_dir / "hero_student.onnx"
    torch.onnx.export(
        model.to("cpu").eval(),
        torch.randn(1, 3, 224, 224),
        str(tmp_onnx),
        input_names=["pixel_values"],
        output_names=["embedding", "logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.save_model(onnx.load(str(tmp_onnx), load_external_data=True), str(onnx_path), save_as_external_data=False)
    tmp_onnx.unlink(missing_ok=True)

    eval_loader = DataLoader(TournamentDistillDataset(records, h2i, spell_imgs, is_train=False), batch_size=64, shuffle=False, num_workers=2)
    model_eval = model.to(device).eval()
    db_vecs = []
    with torch.no_grad():
        for imgs, _, _, _ in eval_loader:
            emb, _ = model_eval(imgs.to(device))
            db_vecs.append(emb.cpu().numpy().astype(np.float16))
    np.savez_compressed(out_dir / "hero_student_db.npz", vectors=np.vstack(db_vecs))
    (out_dir / "hero_student_meta.json").write_text(json.dumps([{"hero": r["hero"], "shape": r["shape"], "source": r["source"], "path": r["rel"]} for r in records], ensure_ascii=False), encoding="utf-8")
    (out_dir / "hero_student_labels.json").write_text(json.dumps(heroes, ensure_ascii=False, indent=2), encoding="utf-8")
    ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("Export complete:", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train distilled MLBB Hero Vision ONNX model")
    p.add_argument("--data-dir", default=None, help="Directory of hero images or hero_ml_bundle.bin")
    p.add_argument("--out-dir", default="./models", help="Output directory for ONNX and index files")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=9600)
    train(p.parse_args())
