import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import math

# ----------------------------
# Configuration
# ----------------------------
BASE = "step_output"
TARGET_W = 480  # All images resized to this width


# ----------------------------
# Load step image
# ----------------------------
def load_step(frame_id, suffix):
    path = os.path.join(BASE, f"{frame_id}_{suffix}.jpg")
    return cv2.imread(path) if os.path.exists(path) else None


# ----------------------------
# Resize image (keep ratio)
# ----------------------------
def resize_keep(img, w=TARGET_W):
    h = int(img.shape[0] * (w / img.shape[1]))
    return cv2.resize(img, (w, h))


# ----------------------------
# Compute PSNR
# ----------------------------
def compute_psnr(img1, img2):
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ----------------------------
# Compute SSIM
# ----------------------------
def compute_ssim(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(g1, g2)


# ----------------------------
# Add text label under image
# ----------------------------
def add_label(img, text):
    label_h = 45
    w = img.shape[1]
    label = np.zeros((label_h, w, 3), dtype=np.uint8)
    cv2.putText(label, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2)
    return np.vstack((img, label))


# ----------------------------
# Main compare function
# ----------------------------
def compare(frame_id):
    print(f"\n🔍 Comparing all steps for frame: {frame_id}\n")

    # File mapping – must match your step_output names
    steps = {
        "1_original":    load_step(frame_id, "1_original"),
        "2_repaired":    load_step(frame_id, "2_repaired"),
        "3_colorized":   load_step(frame_id, "3_colorized"),
        "4_superres":    load_step(frame_id, "4_superres"),
        "5_smooth":      load_step(frame_id, "5_smooth"),
        "6_final":       load_step(frame_id, "6_final"),
        "7_pseudocolor": load_step(frame_id, "7_pseudocolor"),
    }

    names = {
        "1_original":    "Step 1: Original",
        "2_repaired":    "Step 1.5: Luminance Repair",
        "3_colorized":   "Step 2: AI Colorized",
        "4_superres":    "Step 3: Super Resolution",
        "5_smooth":      "Step 4: Temporal Smoothing",
        "6_final":       "Step 5: Final Output",
        "7_pseudocolor": "Pseudocolor (Docs only)",
    }

    # Filter existing images
    images = {k: v for k, v in steps.items() if v is not None}

    if len(images) == 0:
        print("❌ No images found for this frame ID.")
        return

    # Reference size
    first_img = resize_keep(next(iter(images.values())))
    TARGET_H = first_img.shape[0]

    labeled = []
    for key in images:
        img = images[key]
        r = resize_keep(img)
        r = cv2.resize(r, (TARGET_W, TARGET_H))
        labeled.append(add_label(r, names[key]))

    # ---------- Metrics ----------
    if steps["1_original"] is not None and steps["6_final"] is not None:
        print("\n📊 Metrics (Original vs Final Output)")
        o = cv2.resize(resize_keep(steps["1_original"]), (TARGET_W, TARGET_H))
        f = cv2.resize(resize_keep(steps["6_final"]), (TARGET_W, TARGET_H))
        print(f"🔸 PSNR : {compute_psnr(o, f):.2f}")
        print(f"🔸 SSIM : {compute_ssim(o, f):.4f}")
    else:
        print("⚠ Metrics skipped (missing original/final frames)")

    # ---------- Build collage (2 per row, pad last row if odd) ----------
    rows = []
    for i in range(0, len(labeled), 2):
        left = labeled[i]
        if i + 1 < len(labeled):
            right = labeled[i + 1]
        else:
            # Pad with black image of same shape as left
            right = np.zeros_like(left)
        row = np.hstack((left, right))
        rows.append(row)

    collage = np.vstack(rows)

    out_file = f"comparison_{frame_id}.jpg"
    cv2.imwrite(out_file, collage)
    print(f"\n✅ Collage saved → {out_file}")

    cv2.imshow("Comparison View", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    fid = input("Enter Frame ID (e.g., 000123): ").strip()
    compare(fid)
