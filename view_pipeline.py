"""
view_pipeline.py — Canny RVV Project
Stitches every pipeline stage into one collage image for quick visual
sanity-checking. Run this after `make run` / qemu produces output in images/.

Usage:
    python3 view_pipeline.py [width] [height] [original_raw_path]

Defaults to 256x256 and images/rectangle.raw if no args given.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ---------------------------------------------------------------------------
# Config — override via command-line args if needed
# ---------------------------------------------------------------------------
width  = int(sys.argv[1]) if len(sys.argv) > 1 else 256
height = int(sys.argv[2]) if len(sys.argv) > 2 else 256
original_path = sys.argv[3] if len(sys.argv) > 3 else "images/rectangle.raw"

stages = [
    ("Original",     original_path),
    ("Blurred",       "images/blurred.raw"),
    ("Sobel X",       "images/sobel_x.raw"),
    ("Sobel Y",       "images/sobel_y.raw"),
    ("Magnitude L1",  "images/mag_l1.raw"),
    ("Magnitude L2",  "images/mag_l2.raw"),
]

# ---------------------------------------------------------------------------
# Load whatever stages exist, skip missing ones with a warning
# ---------------------------------------------------------------------------
loaded = []
for label, path in stages:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        continue
    raw = np.fromfile(path, dtype=np.uint8)
    expected = width * height
    if raw.size != expected:
        print(f"  [skip] {path}: expected {expected} bytes, got {raw.size}")
        continue
    loaded.append((label, raw.reshape((height, width))))

if not loaded:
    print("No valid stage images found. Run the pipeline first (make run).")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Build the collage: one row, N columns
# ---------------------------------------------------------------------------
n = len(loaded)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))

if n == 1:
    axes = [axes]

for ax, (label, img) in zip(axes, loaded):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(label, fontsize=12)
    ax.axis('off')

fig.suptitle(f"Canny RVV Pipeline — {os.path.basename(original_path)} ({width}x{height})",
             fontsize=14, y=1.02)
plt.tight_layout()

out_name = "pipeline_collage.png"
plt.savefig(out_name, dpi=150, bbox_inches='tight')
print(f"Saved collage to {out_name} — open it from Windows Explorer")
