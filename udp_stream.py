# udp_stream.py
import cv2
import torch
import threading
from ultralytics import YOLO
import torch

# ───────────────────────────────
# 1.  Per-frame processing
# ───────────────────────────────
def yolo_people(frame, conf_threshold: float, model: YOLO):
    """
    Run YOLOv8n on the frame and draw green boxes around *people* (class-id 0).
    """
    # Inference (the model is already on the correct device)
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    for box, conf, cls in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy()):
        if int(cls) != 0:            # keep only the 'person' class
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# ───────────────────────────────
# 2.  Re-usable VideoProcessor
# ───────────────────────────────
class VideoProcessor:
    def __init__(self,
                 processing_function,
                 receive_port=5600,
                 forward_port=5700,
                 forward_host='127.0.0.1',
                 width=1920,
                 height=1080,
                 fps=30,
                 conf_threshold=0.3):

        if not callable(processing_function):
            raise TypeError("processing_function must be callable")

        self.process_func     = processing_function
        self.receive_port     = receive_port
        self.forward_port     = forward_port
        self.forward_host     = forward_host
        self.width            = width
        self.height           = height
        self.fps              = fps
        self.conf_threshold   = conf_threshold

        # ── Device ───────────────────────────────────────────
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")

        # ── Model ────────────────────────────────────────────
        #   yolov8n.pt is tiny (3.2 MB) and trained on COCO (class-0 == person)
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)
        self.model.fuse()      # small speed-up

        # ── GStreamer pipelines ─────────────────────────────
        self.capture_pipeline = (
            f"udpsrc port={self.receive_port} "
            f"! queue "
            f"! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264 "
            f"! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
        )

        self.writer_pipeline = (
            f"appsrc ! videoconvert "
            f"! video/x-raw,format=I420 "
            f"! x264enc speed-preset=ultrafast tune=zerolatency "
            f"! rtph264pay ! udpsink host={self.forward_host} port={self.forward_port}"
        )

        self.capture   = None
        self.writer    = None
        self.is_running = False
        self.thread     = None

    # ────────────────────────────
    # 3.  Public control methods
    # ────────────────────────────
    def start(self):
        if self.is_running:
            print("Already running.")
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("Video processor started.")

    def stop(self):
        print("Stopping video processor…")
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Video processor stopped.")

    # ────────────────────────────
    # 4.  Internal loop
    # ────────────────────────────
    def _run(self):
        self.capture = cv2.VideoCapture(self.capture_pipeline, cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            print("❌ Cannot open incoming stream. Is the sender running?")
            self.is_running = False
            return

        self.writer = cv2.VideoWriter(self.writer_pipeline,
                                      cv2.CAP_GSTREAMER, 0,
                                      self.fps, (self.width, self.height), True)
        if not self.writer.isOpened():
            print("❌ Cannot open outgoing stream.")
            self.is_running = False
            return

        print(f"✅ Streaming {self.width}×{self.height} @ {self.fps} FPS…   Press Q to quit.")

        while self.is_running:
            ret, frame = self.capture.read()
            if not ret:
                print("Stream ended.")
                break

            frame = cv2.resize(frame, (self.width, self.height))
            processed = self.process_func(frame, self.conf_threshold, self.model)
            self.writer.write(processed)
            cv2.imshow("YOLOv8 People Detection", processed)

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()

# ───────────────────────────────
# 5.  Entry-point
# ───────────────────────────────
if __name__ == "__main__":
    vp = VideoProcessor(
        processing_function=yolo_people,
        width=1920, height=1080, fps=30,  # adjust to your OpenHD sender
        receive_port=5600, forward_port=5700
    )

    try:
        vp.start()
        while vp.is_running:
            pass
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        vp.stop()
