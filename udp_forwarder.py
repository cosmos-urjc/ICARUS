import cv2
import numpy as np
import threading
import time

# =============================================================================
#  1. Define Your Processing Functions
# =============================================================================

def draw_green_box(frame):
    """
    Draws a green box in the middle of the frame.
    """
    h, w, _ = frame.shape
    box_w, box_h = 200, 200
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "NVIDIA NVENC", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

# =============================================================================
#  2. The Hardware-Accelerated VideoProcessor Class
# =============================================================================

class VideoProcessor:
    def __init__(self, processing_function, 
                 receive_port=5600, 
                 forward_port=5700, 
                 forward_host='127.0.0.1',
                 width=1280, 
                 height=720, 
                 fps=30):
        if not callable(processing_function):
            raise TypeError("processing_function must be a callable function.")
            
        self.process_func = processing_function
        self.receive_port = receive_port
        self.forward_port = forward_port
        self.forward_host = forward_host
        self.width = width
        self.height = height
        self.fps = fps

        self.capture_pipeline = self._create_capture_pipeline()
        self.writer_pipeline = self._create_writer_pipeline()

        self.capture = None
        self.writer = None
        self.is_running = False
        self.thread = None

    def _create_capture_pipeline(self):
        """
        Pipeline to receive and decode the stream. This still uses a software
        decoder, which is generally fine as decoding is less intensive than encoding.
        """
        return (
            f"udpsrc port={self.receive_port} "
            f"! queue "
            f"! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264 "
            f"! rtph264depay "
            f"! avdec_h264 "
            f"! videoconvert "
            f"! appsink"
        )

    def _create_writer_pipeline(self):
        """
        Creates a robust GStreamer pipeline that uses the NVIDIA hardware encoder (nvh264enc)
        for high performance and low latency.
        """
        # THE KEY CHANGE IS HERE: We replace 'x264enc' with 'nvh264enc'
        # and use presets designed for low-latency hardware encoding.
        return (
            f"appsrc ! queue ! videoconvert ! video/x-raw,format=I420 "
            f"! nvh264enc preset=low-latency-hq bitrate={2000} "
            f"! rtph264pay config-interval=-1 pt=96 "
            f"! udpsink host={self.forward_host} port={self.forward_port}"
        )

    def start(self):
        if self.is_running:
            print("Video processor is already running.")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        print("Video processor started.")

    def _run(self):
        #self.capture = cv2.VideoCapture(self.capture_pipeline, cv2.CAP_GSTREAMER)
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print(f"❌ Error: Cannot open capture pipeline on port {self.receive_port}.")
            self.is_running = False
            return

        self.writer = cv2.VideoWriter(self.writer_pipeline, cv2.CAP_GSTREAMER, 0, self.fps, (self.width, self.height), True)
        if not self.writer.isOpened():
            print("❌ Error: Cannot open writer pipeline. Is 'nvh264enc' available? Check NVIDIA drivers and GStreamer plugins.")
            self.is_running = False
            return
            
        print(f"✅ Pipelines opened with NVIDIA hardware encoding. Processing {self.width}x{self.height} @ {self.fps} FPS video...")
        
        while self.is_running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            processed_frame = self.process_func(frame)
            self.writer.write(processed_frame)

        print("Exiting processing loop.")
        self.capture.release()
        self.writer.release()

    def stop(self):
        print("Stopping video processor...")
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
        print("Video processor stopped.")

# =============================================================================
#  3. How to Run the Class
# =============================================================================

if __name__ == '__main__':
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    VIDEO_FPS = 30

    chosen_effect_function = draw_green_box

    print(f"Starting processor with effect: {chosen_effect_function.__name__}")

    processor = VideoProcessor(
        processing_function=chosen_effect_function,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        fps=VIDEO_FPS
    )
    
    try:
        processor.start()
        while processor.is_running:
            pass
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt.")
    finally:
        processor.stop()