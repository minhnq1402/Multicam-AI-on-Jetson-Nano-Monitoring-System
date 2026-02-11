# 12-Camera Real-time Detection System on Jetson Nano 

Dự án triển khai hệ thống phát hiện vật thể (Lon Bim) thời gian thực trên **12 luồng Camera IP** đồng thời, đẩy hiệu năng của **NVIDIA Jetson Nano** tới giới hạn tối đa.

![Platform](https://img.shields.io/badge/Platform-Jetson%20Nano-green)
![Status](https://img.shields.io/badge/Status-Extreme%20Load-red)
![FPS](https://img.shields.io/badge/FPS-Realtime-blue)

## CẢNH BÁO PHẦN CỨNG (QUAN TRỌNG)
Để chạy được 12 Camera trên Jetson Nano (4GB RAM), bạn **BẮT BUỘC** phải thực hiện các bước sau, nếu không máy sẽ bị treo (Crash):
1. **Tạo RAM ảo (Swap File):** Cần tối thiểu **4GB Swap** (Khuyên dùng 6GB).
2. **Chế độ nguồn:** Bật chế độ hiệu năng cao (Max-N).
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks

Camera Stream: Chỉ sử dụng luồng phụ Sub-stream (VGA 640x480 hoặc thấp hơn). Tuyệt đối không dùng Main-stream (Full HD).

Tính năng
Giám sát diện rộng: Hiển thị lưới 3x4 (12 Camera) trên cùng một màn hình.

Tối ưu bộ nhớ: Sử dụng cơ chế drop=1 và quản lý bộ nhớ đệm chặt chẽ để tránh tràn RAM.

AI Core: YOLOv4-tiny + TensorRT (FP16).

Kết quả: Đã huấn luyện trên 16,000 ảnh , đạt mAP@0.50 ~88%.

🛠 Cài đặt & Sử dụng
Bước 1: Chuẩn bị môi trường

Bước 2: Cài đặt thư viện phụ thuộc
Bash
sudo pip3 install -r requirements.txt
Bước 3: Chuẩn bị Model
Convert model Darknet sang TensorRT engine:

Bash
# Copy file .cfg và .weights vào thư mục yolo/
python3 yolo/yolo_to_onnx.py -m yolov4-tiny-custom
python3 yolo/onnx_to_tensorrt.py -m yolov4-tiny-custom
Bước 4: Cấu hình Camera
Mở file main_12cam.py, chỉnh sửa danh sách RTSP_LINKS. Đảm bảo các link đều là Sub-stream:

Python
RTSP_LINKS = [
    "rtsp://admin:pass@/ch1/sub",
    ...
]
Bước 5: Chạy chương trình
Bash
python3 main_12cam.py
 Hiệu năng (Benchmark)
Thiết bị: Jetson Nano 4GB Dev Kit.

Số lượng Cam: 12.

RAM tiêu thụ: ~2.8 GB / 4.0 GB.

Swap tiêu thụ: ~1.3 GB.

Độ trễ (Latency): < 300ms.

DEMO
https://github.com/user-attachments/assets/487e3974-e5ac-4be8-acf3-37ab80c32a43
