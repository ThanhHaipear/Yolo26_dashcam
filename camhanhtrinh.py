from pathlib import Path
import time

import cv2
from ultralytics import YOLO

# ============================================================
# CAU HINH
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "supbest.pt"
# Neu da co ONNX thi doi thanh:
# MODEL_PATH = BASE_DIR / "supbest.onnx"

CAMERA_INDEX = 1

WINDOW_NAME = "Dashcam Detection FPS Only"

CONF_THRESHOLD = 0.35
IMGSZ = 1280
FLIP_FRAME = False   # Neu khung hinh bi nguoc thi doi True

# ============================================================
# KIEM TRA MODEL
# ============================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

print("======================================")
print("MODEL:", MODEL_PATH)
print("CAMERA_INDEX:", CAMERA_INDEX)
print("CONF_THRESHOLD:", CONF_THRESHOLD)
print("IMGSZ:", IMGSZ)
print("FLIP_FRAME:", FLIP_FRAME)
print("======================================")

# ============================================================
# LOAD MODEL
# ============================================================
model = YOLO(str(MODEL_PATH))
print("Da load model thanh cong.")
print("Class names:", model.names)

# ============================================================
# MO CAMERA
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)

# Neu Windows nhan camera kem, co the thu dong nay thay cho dong tren:
# cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        f"Khong mo duoc camera index = {CAMERA_INDEX}. "
        "Hay thu doi CAMERA_INDEX thanh 1, 2, 3..."
    )

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

print(f"Camera dang mo: {frame_w}x{frame_h}")
print("Nhan 'q' de thoat.")

# ============================================================
# VONG LAP DETECT
# ============================================================
prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        print("Khong doc duoc frame tu camera.")
        break

    if FLIP_FRAME:
        frame = cv2.flip(frame, 1)

    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        verbose=False,
    )

    result = results[0]
    output_frame = result.plot(line_width=2, font_size=0.6)

    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    cv2.putText(
        output_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow(WINDOW_NAME, output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nDa dung boi nguoi dung.")
        break

# ============================================================
# GIAI PHONG
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("Da dong camera.")
