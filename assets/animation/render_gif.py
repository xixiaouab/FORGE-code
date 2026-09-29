from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the FORGE walkthrough from browser frames.")
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timestamps", type=Path)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--crop", type=int, nargs=4, default=[0, 0, 1411, 806])
    args = parser.parse_args()
    if args.timestamps:
        records = json.loads(args.timestamps.read_text())
        records = [(args.frames / Path(r["file"]).name, r["time"]) for r in records if r["time"] < 36000]
    else:
        records = [(p, i * 100) for i, p in enumerate(sorted(args.frames.glob("frame-*.png")))]
    selected = []
    for path, time in records:
        if not selected or time - selected[-1][1] >= 110:
            selected.append((path, time))
    frames = []
    for path, time in selected:
        image = Image.open(path).convert("RGB").crop(tuple(args.crop))
        size = (args.width, round(image.height * args.width / image.width))
        frames.append(image.resize(size, Image.Resampling.LANCZOS))
    samples = [frames[min(len(frames) - 1, round(i * (len(frames) - 1) / 23))] for i in range(24)]
    contact = Image.new("RGB", (args.width * 6, frames[0].height * 4), "white")
    for i, image in enumerate(samples):
        contact.paste(image, ((i % 6) * args.width, (i // 6) * image.height))
    palette = contact.quantize(colors=240, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    durations = [max(20, round((selected[i + 1][1] - time) / 10) * 10) if i + 1 < len(selected) else max(20, round((36000 - time) / 10) * 10) for i, (_, time) in enumerate(selected)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"{len(frames)} frames, {sum(durations) / 1000:.2f} s, {args.output.stat().st_size / 1024 ** 2:.2f} MiB")


if __name__ == "__main__":
    main()
