"""
Project: Extreme 12-Camera Real-time Detection on Jetson Nano
Model: YOLOv4-tiny (TensorRT Optimized)
"""

import cv2
import time
import threading
import numpy as np
from utils.yolo_classes import get_cls_dict
from utils.yolo_with_plugins import TrtYOLO

# --- CẤU HÌNH ---
RTSP_LINKS = [
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
]

MODEL_NAME = 'yolov4-tiny-custom'
CATEGORY_NUM = 1
CONF_THRESH = 0.4
INPUT_SIZE = 416

# Kích thước hiển thị từng ô nhỏ (Giảm xuống để vừa màn hình)
DISPLAY_W, DISPLAY_H = 320, 240 

class RobustGStreamerCamera:
    """Class Camera tối ưu cho tải cao (High Load)"""
    def __init__(self, url, width=416, height=416):
        self.url = url
        self.running = True
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        
        # Pipeline GStreamer: Bỏ videorate, dùng appsink drop=1
        self.pipeline = (
            f"uridecodebin uri={url} source::latency=200 ! "
            "nvvidconv ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1 sync=false max-buffers=1"
        )
        
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"[Error] Không thể kết nối: {url}")

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ret = ret
                else:
                    time.sleep(1) # Chờ 1s nếu mất kết nối
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.ret, self.frame if self.frame is not None else None

    def release(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

def draw_boxes(img, boxes, confs, clss, fps):
    # Vẽ tối giản để tiết kiệm CPU
    for box, conf, cls in zip(boxes, confs, clss):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return img

def main():
    print("[Info] Đang khởi động 12 luồng Camera (Quá trình này mất khoảng 5-10s)...")
    cameras = [RobustGStreamerCamera(url, INPUT_SIZE, INPUT_SIZE) for url in RTSP_LINKS]
    time.sleep(5) # Chờ ổn định

    print(f"[Info] Đang load model {MODEL_NAME}...")
    trt_yolo = TrtYOLO(MODEL_NAME, CATEGORY_NUM)

    print("[Info] Hệ thống sẵn sàng! Nhấn 'q' để thoát.")
    
    fps = 0.0
    tic = time.time()
    frame_count = 0

    while True:
        display_grid = []
        
        # 1. Thu thập và Xử lý từng Cam
        for i, cam in enumerate(cameras):
            ret, frame = cam.read()
            if ret and frame is not None:
                # Detect
                boxes, confs, clss = trt_yolo.detect(frame, CONF_THRESH)
                frame = draw_boxes(frame, boxes, confs, clss, fps)
            else:
                # Màn hình chờ
                frame = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                cv2.putText(frame, f"CAM {i+1}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # Resize để ghép hình
            frame_resized = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            display_grid.append(frame_resized)

        # 2. Ghép lưới 3 hàng x 4 cột
        try:
            row1 = np.hstack(display_grid[0:4])
            row2 = np.hstack(display_grid[4:8])
            row3 = np.hstack(display_grid[8:12])
            final_screen = np.vstack((row1, row2, row3))

            # Hiển thị thông số
            cv2.putText(final_screen, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.imshow("12-Camera Monitor System", final_screen)
        except Exception as e:
            print(f"[UI Error] Lỗi ghép hình: {e}")

        # 3. Tính FPS
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if frame_count == 0 else (0.9 * fps + 0.1 * curr_fps)
        tic = toc
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()