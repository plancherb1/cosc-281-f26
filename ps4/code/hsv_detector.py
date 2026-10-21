"""Provided PS4 color-filter experiment. No implementation is required.

From the course repository root:
    python ps4/code/hsv_detector.py
    python ps4/code/hsv_detector.py ps4/code/photos/daylight.jpg ps4/code/photos/indoor.jpg
    python ps4/code/hsv_detector.py --lighting-demo

Inputs are JPEG or PNG files. Outputs are written to ps4/code/plots.
Keep the thresholds fixed when comparing lighting conditions.
"""
import argparse
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MIN_PIXELS = 20
# Red, orange, yellow, and blue intervals for the supplied images.
THRESHOLDS = {"h_ranges": [(340.0, 65.0), (210.0, 240.0)],
              "s_min": 0.5, "v_min": 0.25}


def rgb_to_hsv(image_rgb):
    """Convert (H, W, 3) RGB to separate hue, saturation, and value grids.

    RGB may use uint8 [0,255] or floating-point [0,1]. Hue is in
    degrees [0,360); saturation and value are in [0,1].
    """
    rgb = np.asarray(image_rgb, dtype=float)
    if rgb.max() > 1:
        rgb = rgb / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = mx - mn
    safe = np.where(delta == 0, 1, delta)
    hue = np.zeros_like(mx)
    mask = (mx == r) & (delta > 0)
    hue[mask] = (60*((g-b)/safe) % 360)[mask]
    mask = (mx == g) & (mx != r) & (delta > 0)
    hue[mask] = (60*((b-r)/safe)+120)[mask]
    mask = (mx == b) & (mx != r) & (mx != g) & (delta > 0)
    hue[mask] = (60*((r-g)/safe)+240)[mask]
    saturation = delta / np.where(mx == 0, 1, mx)
    return hue, saturation, mx


def hsv_mask(image_rgb, h_range, s_min, v_min):
    """Select pixels in a hue interval above saturation/value thresholds."""
    hue, saturation, value = rgb_to_hsv(image_rgb)
    lo, hi = h_range
    in_hue = ((hue >= lo) & (hue <= hi) if lo <= hi
              else (hue >= lo) | (hue <= hi))
    return in_hue & (saturation >= s_min) & (value >= v_min)


def largest_component(mask):
    """Return the largest 4-connected region, its (row, col) mean, and size."""
    from scipy import ndimage
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask), None, 0
    sizes = np.bincount(labels.ravel())[1:]
    label = int(np.argmax(sizes)) + 1
    component = labels == label
    rows, cols = np.nonzero(component)
    return component, (float(rows.mean()), float(cols.mean())), int(sizes.max())


def detect_cone(image_rgb, thresholds=THRESHOLDS):
    """Apply the provided color rule. It does not recognize object shape."""
    mask = np.zeros(np.asarray(image_rgb).shape[:2], dtype=bool)
    for interval in thresholds["h_ranges"]:
        mask |= hsv_mask(image_rgb, interval, thresholds["s_min"],
                         thresholds["v_min"])
    component, centroid, size = largest_component(mask)
    if size < MIN_PIXELS:
        return False, None, np.zeros_like(mask)
    return True, centroid, component


def save_overlay(rgb, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found, centroid, mask = detect_cone(rgb)
    shown = np.asarray(rgb, dtype=float)/255.0
    overlay = shown.copy()
    overlay[mask] = .45*overlay[mask] + .55*np.array([1, 0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(shown)
    axes[0].set_title(title)
    axes[1].imshow(overlay)
    if centroid is not None:
        axes[1].plot(centroid[1], centroid[0], "c+", ms=14, mew=2)
    axes[1].set_title("Selected region" if found else "No region meets thresholds")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path} ({int(mask.sum())} selected pixels)")
    return found, mask


def lighting_examples(rgb):
    """Synthetic changes for a controlled comparison, not a camera model."""
    rgb = np.asarray(rgb, dtype=float)
    return [
        ("original", rgb.astype(np.uint8)),
        ("synthetic_dim", np.clip(.15*rgb, 0, 255).astype(np.uint8)),
        ("synthetic_color_cast",
         np.clip(rgb*np.array([.65, 1.0, .6]), 0, 255).astype(np.uint8)),
    ]


def main():
    from PIL import Image, UnidentifiedImageError
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="*", help="JPEG or PNG image paths")
    parser.add_argument("--lighting-demo", action="store_true",
                        help="compare original, dimmed, and color-shifted samples")
    args = parser.parse_args()
    if args.photos and args.lighting_demo:
        parser.error("Use photo paths or --lighting-demo, not both.")
    data = os.path.join(_HERE, "data", "hsv_frames")
    plots = os.path.join(_HERE, "plots")
    os.makedirs(plots, exist_ok=True)
    if args.lighting_demo:
        image = np.asarray(Image.open(os.path.join(data, "frame_0.png")).convert("RGB"))
        items = lighting_examples(image)
    else:
        paths = args.photos or [os.path.join(data, f"frame_{k}.png") for k in range(3)]
        items = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    items.append((os.path.splitext(os.path.basename(path))[0],
                                  np.asarray(image.convert("RGB"))))
            except (OSError, UnidentifiedImageError):
                parser.error(f"Cannot read {path}. Check the path and use JPEG or PNG.")
    for name, rgb in items:
        save_overlay(rgb, os.path.join(plots, f"hsv_{name}.png"), name)


if __name__ == "__main__":
    main()
