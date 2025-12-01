import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip

input_video = "vid/oldsong.mp4"
output_video = "vid/enhanced_bw.mp4"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print("Frames Loaded:", len(frames))

def enhance_luminance(gray):

    # LOG TRANSFORM
    c = 255 / np.log(1 + np.max(gray))
    log_img = c * np.log(1 + gray)
    log_img = np.uint8(log_img)

    # GAMMA CORRECTION
    gamma = 0.8
    gamma_img = np.uint8(255 * ((log_img / 255) ** gamma))

    # CONTRAST STRETCHING
    p1, p99 = np.percentile(gamma_img, (1, 99))
    stretched = np.uint8(np.clip((gamma_img - p1) * (255 / (p99 - p1)), 0, 255))

    # HISTOGRAM EQUALIZATION
    final = cv2.equalizeHist(stretched)

    return final


processed = []

for frame in frames:
    # Convert BGR → LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Apply composite enhancement on L only
    L2 = enhance_luminance(L)

    # Merge back
    enhanced_lab = cv2.merge((L2, A, B))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    processed.append(enhanced)

print("Enhancement Complete. Rebuilding video...")

h, w, _ = processed[0].shape
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

for f in processed:
    out.write(f)

out.release()

# Restore original audio
clip = VideoFileClip(output_video)
audio = AudioFileClip(input_video)
final = clip.set_audio(audio)
final.write_videofile("vid/enhanced_bw_final.mp4", codec="libx264", audio_codec="aac")

print("DONE! File saved as: vid/enhanced_bw_final.mp4")