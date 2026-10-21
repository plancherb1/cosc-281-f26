"""PS4 Task 1: local image filtering and Sobel responses.

Implement convolve2d and sobel, then run from the course repository root:
    python ps4/code/conv.py
    python ps4/code/autograder.py

The driver supplies the image and writes ps4/code/plots/edges.png.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Provided filters. Apply their displayed coefficients to each image patch.
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def convolve2d(image, kernel, pad=0, stride=1):
    """Return the weighted sum of each image patch using kernel as supplied.

    image has shape (H, W), kernel has odd square shape (K, K).
    pad is the zero border on each side; stride is the spacing between
    successive patch origins. Return a floating-point array of shape
    ((H + 2*pad - K)//stride + 1, (W + 2*pad - K)//stride + 1).

    Suggested steps:
      1. Convert both inputs with np.asarray(..., dtype=float).
      2. Add the border with np.pad(image, pad).
      3. Compute the output shape and allocate np.zeros.
      4. Loop over output (i, j). The patch starts at
         (i*stride, j*stride) in the padded image.
      5. Store np.sum(patch * kernel) in output[i, j].

    Use NumPy and explicit loops, not a library filtering function.
    """
    # TODO: implement the patch-by-patch weighted sums.
    raise NotImplementedError


def sobel(image):
    """Return gx, gy, magnitude, each with the same shape as image.

    Call your convolve2d once per supplied Sobel kernel, with pad=1
    and stride=1. Then compute np.sqrt(gx**2 + gy**2) elementwise.
    """
    # TODO: combine the two directional filter responses.
    raise NotImplementedError


def main():
    """Provided image loading, grayscale conversion, and figure generation."""
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = os.path.join(_HERE, "data", "sim_frame.png")
    rgb = np.asarray(Image.open(frame).convert("RGB"), dtype=float) / 255.0
    gray = 0.299*rgb[..., 0] + 0.587*rgb[..., 1] + 0.114*rgb[..., 2]
    _, _, magnitude = sobel(gray)
    plots = os.path.join(_HERE, "plots")
    os.makedirs(plots, exist_ok=True)
    path = os.path.join(plots, "edges.png")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale input")
    axes[1].imshow(magnitude, cmap="gray")
    axes[1].set_title("Sobel magnitude")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
