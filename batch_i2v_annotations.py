#!/usr/bin/env python3
# Copyright 2024-2025
"""
Batch image-to-video inference over the annotation set.

The script mirrors the single-GPU flow documented in README.md by instantiating
`wan.WanI2V` once and running it for every crop+prompt pair found under the
`./annotations` folder (JSON metadata + `crops/` subdirectory).
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video

TASK_NAME = "i2v-14B"
SIZE_CHOICES = tuple(MAX_AREA_CONFIGS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wan2.1 I2V on every annotation JSON entry.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./Wan2.1-I2V-14B-480P",
        help="Directory that stores the Wan2.1 I2V checkpoints.")
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="./annotations",
        help="Folder that contains the JSON files and the `crops/` subfolder.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/i2v_annotations",
        help="Destination directory for the generated mp4 files.")
    parser.add_argument(
        "--size",
        type=str,
        choices=SIZE_CHOICES,
        default="480*832",
        help="Area target (width*height) as described in README.md.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="Number of frames to sample. Must satisfy 4n+1 (default per README).")
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=40,
        help="Diffusion sampling steps. README uses 40 for I2V.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Flow-matching shift parameter. Defaults to 3.0 for 480p, otherwise 5.0.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        choices=("unipc", "dpm++"),
        default="unipc",
        help="Sampling solver (README defaults to UniPC).")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.")
    parser.add_argument(
        "--no_offload",
        action="store_true",
        help="Keep the DiT / CLIP weights on GPU instead of offloading each step.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Base seed. Set >=0 for deterministic output; each record adds its index.")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device id to use.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate clips even if the corresponding output file already exists.")
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Limit the number of processed annotations (useful for smoke tests).")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        help="Keep the T5 encoder on CPU to reduce GPU memory (matches README flag).")
    return parser.parse_args()


def discover_annotation_files(root: Path) -> List[Path]:
    json_files = sorted(p for p in root.glob("*.json") if p.is_file())
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {root}. Expected annotation metadata.")
    return json_files


def extract_prompt_entries(record: dict) -> List[Tuple[str, str]]:
    """Return (prompt_text, tag) pairs, one per mask (or top-level prompt)."""
    entries: List[Tuple[str, str]] = []
    masks = record.get("masks") or []
    for idx, mask in enumerate(masks, start=1):
        prompt = (mask or {}).get("prompt", "").strip()
        if not prompt:
            continue
        mask_filename = (mask or {}).get("mask_filename")
        tag = Path(mask_filename).stem if mask_filename else f"mask-{idx:03d}"
        entries.append((prompt, tag))

    if not entries:
        top_level_prompt = (record.get("prompt") or "").strip()
        if top_level_prompt:
            entries.append((top_level_prompt, "prompt"))

    if not entries:
        raise ValueError("Prompt information missing from annotation.")
    return entries


def load_crop_image(record: dict, crop_dir: Path) -> Tuple[Path, Image.Image]:
    crop_meta = record.get("crop") or {}
    crop_filename = crop_meta.get("crop_filename")
    if not crop_filename:
        raise ValueError("Annotation has no crop_filename.")
    crop_path = crop_dir / crop_filename
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop image {crop_path} is missing.")
    return crop_path, Image.open(crop_path).convert("RGB")


def iter_annotations(json_files: Iterable[Path]) -> Iterable[Tuple[Path, dict]]:
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError as exc:
            logging.warning("Skipping %s (invalid JSON): %s", json_path, exc)
            continue
        yield json_path, data


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan2.1 inference.")

    annotations_root = Path(args.annotations_dir).resolve()
    crops_dir = annotations_root / "crops"
    if not crops_dir.is_dir():
        raise FileNotFoundError(
            f"{crops_dir} was not found. Expected crop images under annotations.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = discover_annotation_files(annotations_root)
    if args.max_records is not None:
        json_files = json_files[:args.max_records]

    torch.cuda.set_device(args.device)

    cfg = WAN_CONFIGS[TASK_NAME]
    sample_shift = args.sample_shift
    if sample_shift is None:
        sample_shift = 3.0 if args.size in ("480*832", "832*480") else 5.0
    max_area = MAX_AREA_CONFIGS[args.size]
    offload_model = not args.no_offload

    logging.info("Loading Wan2.1 I2V weights from %s", args.ckpt_dir)
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(Path(args.ckpt_dir).resolve()),
        device_id=args.device,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    success, failures = 0, 0
    for index, (json_path, record) in enumerate(iter_annotations(json_files)):
        stem = json_path.stem
        try:
            prompt_entries = extract_prompt_entries(record)
            crop_path, image = load_crop_image(record, crops_dir)
        except (ValueError, FileNotFoundError) as exc:
            logging.warning("Skipping %s: %s", json_path, exc)
            failures += 1
            continue

        for prompt_idx, (prompt, tag) in enumerate(prompt_entries):
            out_file = output_dir / f"{stem}_{tag}.mp4"
            if out_file.exists() and not args.overwrite:
                logging.info("Skipping %s/%s (detected %s).", stem, tag,
                             out_file.name)
                continue

            seed = args.base_seed
            if seed >= 0:
                seed += index * 100 + prompt_idx

            logging.info("Generating %s/%s using %s", stem, tag,
                         crop_path.name)
            try:
                video = wan_i2v.generate(
                    input_prompt=prompt,
                    img=image,
                    max_area=max_area,
                    frame_num=args.frame_num,
                    shift=sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=seed,
                    offload_model=offload_model,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Generation failed for %s/%s: %s", stem, tag,
                                  exc)
                failures += 1
                continue

            cache_video(
                tensor=video[None],
                save_file=str(out_file),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            success += 1
            logging.info("Saved %s", out_file)

    logging.info(
        "Done. Successful clips: %d | Failed: %d | Output dir: %s",
        success,
        failures,
        output_dir,
    )


if __name__ == "__main__":
    main()
