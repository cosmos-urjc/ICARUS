# people_det.py
from ultralytics import YOLO
import cv2
import sys

# ---------- configuration ----------
SOURCE = sys.argv[1] if len(sys.argv) > 1 else '0'   # video file or "0" for webcam
CONF_THRES = 0.25                                    # confidence threshold
DEVICE      = 0                                      # CUDA device index
# ------------------------------------

model = YOLO('yolov8n.pt')           # nano weights (~5 MB)
cap   = cv2.VideoCapture(int(SOURCE) if SOURCE.isdigit() else SOURCE)

if not cap.isOpened():
    raise SystemExit(f"❌  Could not open {SOURCE}")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Predict on the frame, restricting to the COCO “person” class (ID 0)
    # device=0 → GPU; set device="cpu" to force CPU inference
    results = model.predict(
        frame,
        device=DEVICE,
        classes=[0],
        conf=CONF_THRES,
        verbose=False,
        stream=False,   # single image in → single image out
    )

    # Ultralytics returns a list; take the first item and draw it
    annotated = results[0].plot()

    cv2.imshow("YOLOv8n – People detection  |  press q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
