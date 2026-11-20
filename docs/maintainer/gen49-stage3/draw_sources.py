"""Original procedural sources for the frozen four-scene private GEN#49 trial."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, __version__ as pillow_version


def gradient(top, bottom):
    image = Image.new("RGB", (1024, 1024))
    draw = ImageDraw.Draw(image)
    for y in range(1024):
        color = tuple(round(a + (b - a) * y / 1023) for a, b in zip(top, bottom))
        draw.line((0, y, 1023, y), fill=color)
    return image, draw


def curve(points):
    a, b, c, d = points

    def point(i, t):
        return (1 - t) ** 3 * a[i] + 3 * (1 - t) ** 2 * t * b[i] + 3 * (1 - t) * t**2 * c[i] + t**3 * d[i]

    return [(round(point(0, n / 90)), round(point(1, n / 90))) for n in range(91)]


def architecture():
    im, d = gradient((232, 220, 182), (195, 187, 152))
    # The long triangular trusses have vertices away from the seam.
    d.rectangle((0, 312, 1023, 815), fill="#869da0")
    for y in range(322, 804, 48):
        d.line((0, y, 1023, y), fill="#bdc4b8", width=3)
    for i in range(-1, 5):
        x = 114 + 228 * i
        d.polygon([(x, 675), (x + 114, 389), (x + 228, 675)], fill="#536967")
        d.polygon([(x + 17, 659), (x + 114, 415), (x + 211, 659)], fill="#9cac9f")
        d.line((x + 114, 389, x + 114, 673), fill="#56645e", width=8)
        d.line((x + 24, 618, x + 204, 618), fill="#555e56", width=6)
    for y, h in [(382, 13), (505, 13), (670, 21), (717, 12)]:
        d.rectangle((0, y, 1023, y + h), fill="#3f4e50")
        d.line((0, y, 1023, y), fill="#f1dfae", width=3)
    for x in (78, 357, 655, 937):
        d.polygon([(x - 24, 721), (x + 24, 721), (x + 13, 899), (x - 13, 899)], fill="#77685b")
        d.rectangle((x - 31, 718, x + 31, 734), fill="#655b51")
    for y in range(842, 1024):
        d.line((0, y, 1023, y), fill=(99 - (y - 842) // 13, 131 - (y - 842) // 14, 131 - (y - 842) // 14))
    rng = random.Random(4101)
    for _ in range(240):
        x, y = rng.randrange(1024), rng.randrange(844, 1024)
        d.line((x, y, x + rng.randrange(5, 60), y), fill=rng.choice(["#b2ba9e", "#738d89"]), width=2)
    return im


def organic():
    im, d = gradient((40, 61, 52), (19, 42, 45))
    rng = random.Random(4201)
    for _ in range(1200):
        x, y = rng.randrange(1024), rng.randrange(1024)
        radius = rng.choice([1, 2, 3, 5])
        d.ellipse((x, y, x + radius, y + radius), fill=rng.choice(["#3b5446", "#465a47", "#213c3e"]))

    # Alternating toothed leaflets grow from a curving stem, with visible veins.
    def frond(start, end, bend, spacing, colors):
        sx, sy = start
        ex, ey = end

        def center(t):
            return (sx + (ex - sx) * t + bend * math.sin(math.pi * t), sy + (ey - sy) * t)

        for j in range(spacing):
            for side in (-1, 1):
                t = (j + 1 + (0.34 if side > 0 else 0)) / (spacing + 2)
                x, y = center(t)
                dx = ex - sx + bend * math.pi * math.cos(math.pi * t)
                dy = ey - sy
                length = math.hypot(dx, dy)
                tx, ty = dx / length, dy / length
                nx, ny = -ty * side, tx * side
                spread = 33 + 128 * (math.sin(math.pi * t) ** 0.75)
                tip = (x + (nx * 0.82 + tx * 0.68) * spread, y + (ny * 0.82 + ty * 0.68) * spread)
                lx, ly = tip[0] - x, tip[1] - y

                def edge(k, sign, x=x, y=y, lx=lx, ly=ly, spread=spread):
                    u = k / 16
                    width = (12 + 3 * (k % 2)) * math.sin(math.pi * u) * sign
                    return (x + lx * u - ly * width / spread, y + ly * u + lx * width / spread)

                d.polygon(
                    [edge(k, 1) for k in range(17)] + [edge(k, -1) for k in reversed(range(17))],
                    fill=colors[(j + (side == 1)) % 2],
                )
                d.line((x, y, *tip), fill=colors[2], width=2)
                for k in (4, 7, 10, 13):
                    u = (k - 2) / 16
                    for s in (-1, 1):
                        d.line([(x + lx * u, y + ly * u), edge(k, s)], fill=colors[2], width=1)
        d.line([center(k / 150) for k in range(151)], fill=colors[2], width=5)

    frond((-60, 570), (345, -24), 65, 17, ("#4d755b", "#5b7d5c", "#85916a"))
    frond((569, 1100), (1050, 298), -56, 20, ("#557c64", "#638a69", "#8ca37b"))
    frond((117, 1004), (885, 14), 12, 25, ("#879e62", "#9ba966", "#c2be7a"))
    return im


def lettering():
    im, d = gradient((213, 184, 147), (178, 132, 99))
    d.rounded_rectangle((68, 172, 955, 852), radius=42, fill="#334d59", outline="#574d45", width=12)
    d.rounded_rectangle((86, 190, 937, 834), radius=33, outline="#d6c49c", width=6)
    for x in (119, 905):
        for y in (224, 800):
            d.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#d8cbb1", outline="#172f38", width=3)
            d.line((x - 4, y + 4, x + 4, y - 4), fill="#4d5d61", width=2)
    # Original geometric monoline glyphs. The E's middle bar intersects x=y=512.
    letters = {
        "T": [[(0, 0), (58, 0)], [(29, 0), (29, 100)]],
        "I": [[(5, 0), (51, 0)], [(28, 0), (28, 100)], [(5, 100), (51, 100)]],
        "D": [[(0, 100), (0, 0), (30, 0), (54, 17), (58, 35), (58, 65), (54, 83), (30, 100), (0, 100)]],
        "E": [[(58, 0), (0, 0), (0, 100), (58, 100)], [(0, 50), (55, 50)]],
        "0": [[(14, 0), (43, 0), (57, 14), (57, 85), (43, 100), (14, 100), (0, 85), (0, 14), (14, 0)]],
        "7": [[(0, 0), (59, 0), (15, 100)]],
    }
    for x, char in zip((156, 259, 361, 469, 641, 752), "TIDE07"):
        for segment in letters[char]:
            d.line([(x + a * 1.42, 433 + b * 1.58) for a, b in segment], fill="#f4e4b9", width=12, joint="curve")
    d.line((168, 684, 854, 684), fill="#ecd9ac", width=3)
    for x in range(176, 854, 10):
        d.line((x, 684, x, 705 if (x - 176) % 50 == 0 else 694), fill="#ecd9ac", width=2)
    for x in range(319, 729, 32):
        d.arc((x, 301, x + 32, 334), start=0, end=180, fill="#9abfbd", width=3)
    return im


def subject():
    im, d = gradient((78, 131, 141), (36, 82, 100))
    rng = random.Random(4401)
    for _ in range(125):
        x, y = rng.randrange(-80, 1024), rng.randrange(1024)
        w, h = rng.randrange(45, 165), rng.randrange(9, 32)
        d.arc((x, y, x + w, y + h), start=15, end=161, fill=rng.choice(["#619198", "#42798a", "#7fa3a2"]), width=2)

    # One continuous diagonal koi. The pale-and-red torso occupies the centre.
    def shape(curves, fill, outline="#514c49"):
        pts = [p for part in curves for p in curve(part)]
        d.polygon(pts, fill=fill)
        d.line(pts + [pts[0]], fill=outline, width=3)

    shape(
        [
            [(690, 441), (770, 390), (766, 225), (861, 176)],
            [(861, 176), (850, 234), (875, 290), (827, 344)],
            [(827, 344), (870, 323), (921, 327), (981, 297)],
            [(981, 297), (945, 399), (807, 428), (690, 441)],
        ],
        "#d7c69b",
    )
    shape(
        [
            [(454, 467), (421, 402), (393, 384), (386, 325)],
            [(386, 325), (471, 332), (536, 376), (552, 448)],
            [(552, 448), (505, 472), (481, 483), (454, 467)],
        ],
        "#e3cba0",
    )
    shape(
        [
            [(495, 573), (546, 604), (574, 667), (564, 708)],
            [(564, 708), (634, 678), (643, 593), (611, 524)],
            [(611, 524), (560, 512), (525, 562), (495, 573)],
        ],
        "#dbc69b",
    )
    shape(
        [
            [(152, 727), (144, 641), (280, 497), (454, 434)],
            [(454, 434), (601, 376), (660, 402), (761, 358)],
            [(761, 358), (787, 378), (784, 398), (764, 423)],
            [(764, 423), (633, 478), (619, 556), (527, 615)],
            [(527, 615), (394, 711), (214, 820), (152, 727)],
        ],
        "#f2dfb7",
    )
    # Patches sit inside the curved silhouette and provide countable anchors.
    for box, fill in [
        ((233, 603, 329, 680), "#c86148"),
        ((352, 505, 453, 579), "#b9493a"),
        ((457, 465, 558, 542), "#c55d47"),
        ((576, 430, 643, 477), "#a74d40"),
    ]:
        d.ellipse(box, fill=fill)
    for n in range(6):
        x, y = 347 + n * 41, 610 - n * 24
        for k in range(3):
            d.arc((x + k * 9, y + k * 10, x + 16 + k * 9, y + 16 + k * 10), start=345, end=165, fill="#ad8c74", width=2)
    d.ellipse((211, 654, 239, 681), fill="#aa8568", outline="#5b5049", width=3)
    d.ellipse((218, 661, 230, 674), fill="#303d43")
    d.ellipse((219, 662, 222, 665), fill="#f8e8cb")
    d.line(curve([(153, 717), (130, 721), (116, 737), (106, 746)]), fill="#dcc9a2", width=3)
    d.line(curve([(166, 749), (141, 752), (133, 777), (114, 774)]), fill="#dcc9a2", width=3)
    return im


SCENES = [
    (
        architecture,
        4101,
        "architectural illustration, steel truss bridge in side elevation, straight horizontal dark girders, repeated triangular braces, muted blue and warm gray, water below",
        "Follow straight girders and repeating diagonal braces across both centre lines; check for broken lines, doubling, invented braces or missing spans.",
    ),
    (
        organic,
        4201,
        "botanical illustration, overlapping diagonal fern fronds, rows of individual serrated green leaflets with delicate central veins against a dark green ground",
        "Follow the pale diagonal frond through the centre; compare leaflet count, pointed teeth, alternating bases and veins, including newly invented or blurred detail.",
    ),
    (
        lettering,
        4301,
        "vintage enamel tide gauge sign, readable cream geometric lettering TIDE 07 on dark blue, narrow ruler ticks below, decorative wave and four corner screws",
        "Read TIDE 07, including the centred E crossbar, compare narrow ruler ticks, four screws and wave detail; count changes as fidelity failures even with smooth seams.",
    ),
    (
        subject,
        4401,
        "illustrated koi fish in blue water, one cream fish swimming to the lower left, four red patches, diagonal body, two pale fins and a forked tail",
        "Trace one fish outline across both central boundaries, four patches, scale arcs, eye, two fins and tail fork; distinguish new detail from changed anatomy.",
    ),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    output = parser.parse_args().directory
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "gen49-owned-fixtures-v1",
        "provenance": "Four original procedural illustrations drawn with this Python source, no external image or font assets.",
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "pillow": pillow_version,
        "preparation": "1024-square RGB to 2048-square RGB, Pillow Lanczos",
        "frozen_native_boxes_2048": {
            "vertical": [880, 256, 1168, 1792],
            "horizontal": [256, 880, 1792, 1168],
            "centre": [768, 768, 1280, 1280],
        },
        "scenes": [],
    }
    for draw, seed, prompt, criteria in SCENES:
        im = draw()
        path = output / (draw.__name__ + ".png")
        im.save(path)
        prepared = im.resize((2048, 2048), Image.Resampling.LANCZOS)
        manifest["scenes"].append(
            {
                "name": draw.__name__,
                "filename": path.name,
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": "low quality, blurry",
                "rgb_sha256": digest(im.tobytes()),
                "prepared_rgb_sha256": digest(prepared.tobytes()),
                "png_sha256": digest(path.read_bytes()),
                "fidelity_criteria": criteria,
            }
        )
        prepared.close()
        im.close()
    # The original manifest records the generator at ad0912f. Preserve its
    # bytes after verifying every current PNG, RGB proof and conditioning field.
    frozen_bytes = Path(__file__).with_name("manifest.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    for key in manifest.keys() - {"generator_sha256"}:
        if manifest[key] != frozen[key]:
            raise ValueError(f"Generated fixture does not reproduce the frozen {key}.")
    (output / "manifest.json").write_bytes(frozen_bytes)


if __name__ == "__main__":
    main()
