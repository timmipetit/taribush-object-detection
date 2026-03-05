import cv2
import numpy as np
import time
from datetime import datetime
import subprocess
import os
import tflite_runtime.interpreter as tflite

# === CONFIG ===
USB_MOUNT_BASE = "/media"  # Raspberry Pi OS mounts USB sticks here
CONFIDENCE_THRESHOLD = 0.5
DETECTION_TIME_THRESHOLD = 3   # Seconds of detection before playing
LOSS_TIME_THRESHOLD = 3        # Seconds of no detection before stopping


def find_usb_content():
    """Find a USB stick containing a .tflite model and labels.txt. Returns (stick_path, model_path) or (None, None)."""
    for user_dir in os.listdir(USB_MOUNT_BASE):
        user_path = os.path.join(USB_MOUNT_BASE, user_dir)
        if not os.path.isdir(user_path):
            continue
        for stick in os.listdir(user_path):
            stick_path = os.path.join(user_path, stick)
            if not os.path.isdir(stick_path):
                continue
            for f in os.listdir(stick_path):
                if f.endswith(".tflite"):
                    return (stick_path, os.path.join(stick_path, f))
    return (None, None)


def play_audio(audio_path):
    return subprocess.Popen([
        "mpv", audio_path,
        "--loop",            # Loop forever
        "--no-video",        # Audio only
        "--really-quiet",    # Suppress logs
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_audio(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


# === Find USB stick ===
print("Looking for USB stick with a .tflite model...")
usb_path, model_path = find_usb_content()
if not usb_path:
    print("No USB stick found with a .tflite model. Please insert a USB stick containing:")
    print("  - a .tflite model file")
    print("  - labels.txt")
    print("  - scene1.mp3, scene2.mp3, ... (matching the labels)")
    exit(1)

print(f"Found content on: {usb_path}")
print(f"Using model: {os.path.basename(model_path)}")

# === Load labels from Teachable Machine export ===
with open(os.path.join(usb_path, "labels.txt"), "r") as f:
    # Teachable Machine format: "0 scene1", "1 scene2", etc.
    labels = [line.strip().split(maxsplit=1)[-1] for line in f.readlines() if line.strip()]

# === Map labels to mp3 files (scene1 -> scene1.mp3, etc.) ===
scenes = {}
for label in labels:
    audio_file = os.path.join(usb_path, f"{label}.mp3")
    if os.path.isfile(audio_file):
        print(f"  Found {label}.mp3")
        scenes[label] = audio_file
    else:
        print(f"  No {label}.mp3 found, skipping")

print(f"Loaded {len(scenes)} scene(s) with audio: {', '.join(scenes.keys())}")

# === Load TFLite model ===
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# === Camera input ===
cap = cv2.VideoCapture(0)

detection_start = None   # (scene_name, timestamp) for the scene we're tracking
last_active = time.time()  # last time any known scene was detected
audio_proc = None
current_scene = None


def detect_scene():
    """Classify the current frame. Returns (label, confidence)."""
    ret, frame = cap.read()
    if not ret:
        return (None, 0)

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Center crop to square before resizing to avoid squishing
    h, w, _ = frame_rgb.shape
    size = min(h, w)
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    cropped = frame_rgb[start_y:start_y+size, start_x:start_x+size]

    img = cv2.resize(cropped, (width, height))
    input_data = np.expand_dims(img, axis=0)

    # Normalize based on model type
    input_type = input_details[0]['dtype']
    if input_type == np.float32:
        # Teachable Machine (MobileNet) expects -1.0 to 1.0
        input_data = (input_data.astype(np.float32) / 127.5) - 1.0
    else:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    top_index = int(np.argmax(output_data))
    if input_type == np.uint8 or output_data[top_index] > 1.0:
        confidence = output_data[top_index] / 255.0
    else:
        confidence = float(output_data[top_index])
    return (labels[top_index], confidence)


print("Looking...")

while True:
    (detected, confidence) = detect_scene()
    now = time.time()

    known_label = detected and confidence > CONFIDENCE_THRESHOLD

    if known_label:
        # Track which specific scene we're seeing consistently
        if detection_start is None or detection_start[0] != detected:
            detection_start = (detected, now)
        # Any known scene keeps the "last activity" alive
        last_active = now
    else:
        detection_start = None

    # Switch to new scene if detected consistently for long enough
    if detection_start:
        scene, start_time = detection_start
        if scene != current_scene and now - start_time >= DETECTION_TIME_THRESHOLD:
            stop_audio(audio_proc)
            audio_proc = None
            if scene in scenes:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected {scene} for {DETECTION_TIME_THRESHOLD}s, playing {scenes[scene]}")
                audio_proc = play_audio(scenes[scene])
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected {scene} for {DETECTION_TIME_THRESHOLD}s (no audio)")
            current_scene = scene

    # Stop audio if no known scene detected for LOSS_TIME_THRESHOLD
    if current_scene and now - last_active > LOSS_TIME_THRESHOLD:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Lost sight of {current_scene} for {LOSS_TIME_THRESHOLD}s, stopping audio")
        stop_audio(audio_proc)
        audio_proc = None
        current_scene = None
        detection_start = None

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()

