"""Check the 500 released Calipso PNGs and, optionally, private originals."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image


ALPHA_KEY = "ISAR_model_alpha_zlib_base64_v1"


def embedded_mask(image: Image.Image) -> np.ndarray:
    """Decode the lossless support mask embedded in a release PNG."""
    encoded = image.info.get(ALPHA_KEY)
    if not isinstance(encoded, str):
        raise AssertionError("PNG is missing its embedded support mask")
    try:
        data = zlib.decompress(base64.b64decode(encoded, validate=True))
    except (ValueError, zlib.error) as exc:
        raise AssertionError("invalid embedded support mask") from exc
    if len(data) != image.width * image.height:
        raise AssertionError("embedded support mask has the wrong size")
    return np.frombuffer(data, dtype=np.uint8).reshape(image.height, image.width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path,
                        help="Override the scene's masked-image directory")
    parser.add_argument("--original-dir", type=Path,
                        help="Optional private, pre-blackening model-input directory")
    parser.add_argument("--original-scene", type=Path,
                        help="Optional private scene whose cameras.json maps original filenames")
    args = parser.parse_args()

    scene = args.scene.resolve()
    frames = json.loads((scene / "cameras.json").read_text(encoding="utf-8"))["frames"]
    if len(frames) != 500:
        raise AssertionError(f"expected 500 frames, got {len(frames)}")
    image_dir = args.image_dir.resolve() if args.image_dir else scene / "image"
    names = [frame["filename"] for frame in frames]
    if names != [f"{index}.png" for index in range(1, 501)]:
        raise AssertionError("camera filenames must be 1.png through 500.png in order")
    if len(set(names)) != 500 or {p.name for p in image_dir.glob("*.png")} != set(names):
        raise AssertionError("PNG names do not match the 500 camera frames")
    original_frames = None
    if args.original_scene:
        original_frames = json.loads(
            (args.original_scene / "cameras.json").read_text(encoding="utf-8")
        )["frames"]
        if len(original_frames) != 500:
            raise AssertionError("private scene does not have 500 camera frames")

    for name in names:
        with Image.open(image_dir / name) as image:
            if image.mode != "RGB" or image.size != (512, 512):
                raise AssertionError(f"wrong mode or size: {name}")
            masked = np.asarray(image).copy()
            alpha = embedded_mask(image)
        if np.any(masked[alpha == 0] != 0):
            raise AssertionError(f"nonblack RGB outside alpha support: {name}")
        if args.original_dir:
            index = int(Path(name).stem)
            if original_frames is not None:
                original_name = original_frames[index - 1]["filename"]
            else:
                original_name = name
            with Image.open(args.original_dir / original_name) as image:
                original = np.asarray(image.convert("RGBA"))
            if not np.array_equal(alpha, original[..., 3]):
                raise AssertionError(f"alpha changed: {name}")
            if not np.array_equal(masked[alpha > 0],
                                  original[..., :3][alpha > 0]):
                raise AssertionError(f"model-relevant RGB changed: {name}")
    print(f"Verified {len(names)} opaque RGB images and embedded masks; private comparison={bool(args.original_dir)}")


if __name__ == "__main__":
    main()
