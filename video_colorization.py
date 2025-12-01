from moviepy.editor import VideoFileClip, AudioFileClip 
import argparse, os, cv2, shutil
import matplotlib.pyplot as plt
from colorizers import *   # eccv16, siggraph17
from PIL import Image
import numpy as np


# ============================================================
#           STEP OUTPUT SAVING FUNCTION (DOCUMENTATION)
# ============================================================

os.makedirs("step_output", exist_ok=True)

def save_step(frame, frame_id, step_name):
    """Save intermediate step outputs for documentation only"""
    cv2.imwrite(f"step_output/{frame_id}_{step_name}.jpg", frame)


# ============================================================
#       COMPOSITE LUMINANCE ENHANCEMENT (LOG + GAMMA + STRETCH)
# ============================================================

def composite_luminance(L_channel):
    """
    Apply composite enhancement on luminance:
    1) Log transform
    2) Gamma correction
    3) Contrast stretching (1–99 percentile)
    4) Return 0–255 uint8 image
    """
    L = L_channel.astype(np.float32) / 255.0  # normalize to 0–1

    # 1. LOG TRANSFORM (boost dark regions)
    L_log = np.log1p(L)          # log(1 + L)
    L_log /= np.log1p(1.0)       # normalize by log(2), keeps in 0–1-ish

    # 2. GAMMA CORRECTION (tone control)
    gamma = 0.9                  # <1 → slightly brighten midtones
    L_gamma = np.power(L_log, gamma)

    # 3. CONTRAST STRETCH (avoid washed-out look)
    p1, p99 = np.percentile(L_gamma, (1, 99))
    if p99 - p1 < 1e-6:
        L_stretch = L_gamma
    else:
        L_stretch = (L_gamma - p1) / (p99 - p1)
    L_stretch = np.clip(L_stretch, 0, 1)

    # Back to 0–255
    L_out = (L_stretch * 255).astype(np.uint8)
    return L_out


# ============================================================
#                     ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="eccv16",
                    choices=["eccv16", "siggraph17"],
                    help="Choose colorization model")
args = parser.parse_args()


# ============================================================
#                STEP 1: FRAME EXTRACTION + RAW PSEUDOCOLOR
# ============================================================

def step1():
    print("\n[STEP 1] Extracting frames...\n")

    for folder in ["vid_out", "bw_vid_out", "sr_out", "smooth_out", "ivp_out"]:
        try:
            shutil.rmtree(folder)
        except:
            pass

    os.makedirs('vid_out', exist_ok=True)

    cap = cv2.VideoCapture("vid/oldsong.mp4")
    count = 0
    success, frame = cap.read()

    while success:
        frame_id = str(count).zfill(6)
        cv2.imwrite(f"vid_out/{frame_id}.jpg", frame)

        # STEP OUTPUT 1: Original Frame
        save_step(frame, frame_id, "1_original")

        # Raw grayscale pseudocolor (DOC ONLY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pseudo_raw = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        save_step(pseudo_raw, frame_id, "0_pseudocolor_raw")

        success, frame = cap.read()
        count += 1

    cap.release()
    print(f"Extracted {count} frames.")


# ============================================================
#                STEP 2: COLORIZATION (UNCHANGED)
# ============================================================

def step2():
    print("\n[STEP 2] Colorizing frames...\n")

    os.makedirs("bw_vid_out", exist_ok=True)

    # Load selected model
    colorizer = eccv16(pretrained=True).eval() if args.model == "eccv16" else siggraph17(pretrained=True).eval()

    def colorize(input_path, output_path):
        img = load_img(input_path)                     # colorizers' utility
        L_orig, L_rs = preprocess_img(img, HW=(256, 256))
        out_img = postprocess_tens(L_orig, colorizer(L_rs).cpu())

        # Save as RGB image [0–1] using matplotlib
        plt.imsave(output_path, out_img)

        # Also save for documentation as BGR
        bgr = cv2.cvtColor((out_img * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
        frame_id = os.path.basename(output_path).split('.')[0]
        save_step(bgr, frame_id, "2_colorized")

    for img in sorted(os.listdir("vid_out")):
        colorize(f"vid_out/{img}", f"bw_vid_out/{img}")

    print("Colorization complete.")


# ============================================================
#             STEP 2.5 SUPER RESOLUTION
# ============================================================

def super_resolution():
    print("\n[STEP 2.5] Super Resolution...\n")

    os.makedirs("sr_out", exist_ok=True)

    for file in sorted(os.listdir("bw_vid_out")):
        img = cv2.imread(f"bw_vid_out/{file}")
        sr = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"sr_out/{file}", sr)

        frame_id = file.split('.')[0]
        save_step(sr, frame_id, "3_superres")

    print("Super-resolution complete.")


# ============================================================
#                STEP 2.6 TEMPORAL SMOOTHING
# ============================================================

def temporal_smoothing():
    print("\n[STEP 2.6] Temporal Smoothing...\n")

    os.makedirs("smooth_out", exist_ok=True)
    frames = sorted(os.listdir("sr_out"))

    for i in range(1, len(frames)-1):
        prev = cv2.imread(f"sr_out/{frames[i-1]}")
        curr = cv2.imread(f"sr_out/{frames[i]}")
        nxt = cv2.imread(f"sr_out/{frames[i+1]}")

        smoothed = (prev * 0.25 + curr * 0.5 + nxt * 0.25).astype(np.uint8)

        cv2.imwrite(f"smooth_out/{frames[i]}", smoothed)

        frame_id = frames[i].split('.')[0]
        save_step(smoothed, frame_id, "4_smooth")

    print("Temporal smoothing complete.")


# ============================================================
#      STEP 2.7 IVP ENHANCEMENTS (COMPOSITE LUMINANCE + HE)
# ============================================================

def ivp_enhancements():
    print("\n[STEP 2.7] Applying IVP + Composite Enhancements...\n")

    os.makedirs("ivp_out", exist_ok=True)

    for file in sorted(os.listdir("smooth_out")):
        img = cv2.imread(f"smooth_out/{file}")

        # 1. Sharpening (spatial enhancement)
        sharpen_kernel = np.array([[0,-1,0],
                                   [-1,5,-1],
                                   [0,-1,0]])
        sharpened = cv2.filter2D(img, -1, sharpen_kernel)

        # 2. Go to LAB for luminance-based enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 3. Composite luminance enhancement: log + gamma + contrast stretch
        l_comp = composite_luminance(l)

        # 4. Final Histogram Equalization on enhanced L
        l_eq = cv2.equalizeHist(l_comp)

        # 5. Merge back and convert to BGR
        lab_eq = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Save to ivp_out
        cv2.imwrite(f"ivp_out/{file}", enhanced)

        frame_id = file.split('.')[0]
        save_step(enhanced, frame_id, "5_ivp_composite")

        # 6. PSEUDOCOLOR ONLY FOR DOCUMENTATION (NOT USED IN VIDEO)
        pseudo = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        save_step(pseudo, frame_id, "6_pseudocolor")

    print("IVP + Composite Enhancements complete.")


# ============================================================
#                STEP 3 REBUILD FINAL VIDEO
# ============================================================

def step3():
    print("\n[STEP 3] Rebuilding Video...\n")

    path = "ivp_out"
    files = sorted(os.listdir(path))

    first = cv2.imread(f"{path}/{files[0]}")
    h, w, c = first.shape

    out = cv2.VideoWriter("vid/mygeneratedvideo.avi",
                          cv2.VideoWriter_fourcc(*'XVID'),
                          24, (w, h))

    for f in files:
        out.write(cv2.imread(f"{path}/{f}"))

    out.release()
    print("Video rebuilt.")


# ============================================================
#                STEP 4 ADD ORIGINAL AUDIO
# ============================================================

def step4():
    print("\n[STEP 4] Adding Audio...\n")

    clip = VideoFileClip("vid/mygeneratedvideo.avi")
    audio = AudioFileClip("vid/oldsong.mp4")

    final = clip.set_audio(audio)
    final.write_videofile("vid/coloured.mp4",
                          codec="libx264",
                          audio_codec="aac")

    print("Final video saved → vid/coloured.mp4")


# ============================================================
#                        MAIN RUN
# ============================================================

if __name__ == "__main__":
    step1()
    step2()
    super_resolution()
    temporal_smoothing()
    ivp_enhancements()
    step3()
    step4()