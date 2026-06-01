# Real-time Multi-Camera RTSP Edge AI Monitoring System on Jetson Nano

This project implements a real-time multi-camera video monitoring system on NVIDIA Jetson Nano.

The system receives multiple RTSP streams from IP cameras, processes them through a GStreamer pipeline, performs AI-based object detection using YOLOv4-tiny optimized with TensorRT FP16, and displays all camera streams in a low-latency monitoring screen.

The main purpose of this project is not only object detection, but also building a practical edge video processing pipeline that combines:

* Multi-camera RTSP stream input
* Low-latency video decoding with GStreamer
* Real-time AI inference on an edge device
* Batched inference to reduce GPU bottlenecks
* Memory and latency optimization under limited hardware resources
* Grid-based monitoring UI for multiple camera streams

---

## 1. Project Overview

In real-world monitoring systems, multiple IP cameras often need to be processed at the same time.
However, running AI inference separately for each camera can easily overload the GPU, especially on resource-constrained edge devices such as Jetson Nano.

This project was developed to solve that problem by combining:

* RTSP stream processing
* GStreamer-based low-latency video input
* Multi-threaded camera capture
* Latest-frame buffering
* YOLOv4-tiny object detection
* TensorRT FP16 optimization
* Batched inference for multiple cameras
* Real-time multi-camera visualization

The system is designed to prioritize real-time responsiveness and stable operation rather than processing every single frame.

---

## 2. Key Features

* Real-time processing of multiple IP camera RTSP streams
* GStreamer-based video pipeline for low-latency frame acquisition
* YOLOv4-tiny object detection optimized with TensorRT FP16
* Batched inference to reduce GPU bottlenecks when processing many cameras
* Latest-frame strategy to avoid latency accumulation
* Multi-threaded camera capture to prevent one unstable stream from blocking the whole system
* 3x4 grid monitoring screen for visualizing multiple camera streams
* Frame dropping and buffer control to reduce latency and RAM usage
* Designed for resource-constrained edge devices such as Jetson Nano

---

## 3. System Architecture

```text
IP Cameras
   │
   │ RTSP Streams
   ▼
GStreamer Pipeline
   │
   │ uridecodebin / nvvidconv / appsink
   ▼
Latest Frame Buffer
   │
   │ drop old frames to avoid latency accumulation
   ▼
Batch Frame Collector
   │
   │ collect latest frames from active cameras
   ▼
YOLOv4-tiny + TensorRT FP16
   │
   │ batched edge AI inference
   ▼
Detection Result Mapping
   │
   │ split results by camera index
   ▼
OpenCV Visualization
   │
   │ 3x4 multi-camera grid
   ▼
Real-time Monitoring Screen
```

---

## 4. Technology Stack

| Category           | Technologies                              |
| ------------------ | ----------------------------------------- |
| Programming        | Python                                    |
| Computer Vision    | OpenCV, YOLOv4-tiny                       |
| Video Streaming    | RTSP, GStreamer                           |
| AI Optimization    | TensorRT FP16, CUDA                       |
| Inference Strategy | Batched Inference, Latest-frame Buffering |
| Edge Device        | NVIDIA Jetson Nano                        |
| OS                 | Ubuntu / JetPack                          |
| Development Tools  | Git, VSCode                               |

---

## 5. GStreamer Pipeline

The system uses a GStreamer pipeline to receive RTSP streams and reduce latency.

```python
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
```

Important optimization points:

* `source::latency=200`: controls RTSP buffering latency
* `appsink drop=1`: drops old frames to avoid delay accumulation
* `sync=false`: prevents display synchronization delay
* `max-buffers=1`: limits frame buffering to reduce memory usage

This design is important for real-time monitoring because old frames are no longer useful when the system needs to display the latest camera status.

---

## 6. Batched Inference Optimization

When multiple RTSP cameras are processed simultaneously, running AI inference separately for each camera can create a GPU bottleneck.

Each camera continuously generates frames. If every frame from every camera is sent to the model independently, the system produces many small inference requests. This increases GPU scheduling overhead and reduces overall throughput.

To reduce this bottleneck, this project uses a batched inference strategy.

Instead of running inference camera by camera, the system collects the latest available frames from multiple camera streams and combines them into a single batch. The batch is then passed to the YOLOv4-tiny TensorRT engine in one inference call. After inference, the detection results are separated and mapped back to the corresponding camera streams.

```text
Multiple RTSP Cameras
        │
        ▼
Latest Frame Buffer
        │
        ▼
Batch Frame Collector
        │
        ▼
YOLOv4-tiny TensorRT Batched Inference
        │
        ▼
Split Detection Results by Camera
        │
        ▼
Multi-camera Monitoring Display
```

Key design points:

* Each camera capture thread stores only the latest frame.
* Old frames are dropped to prevent latency accumulation.
* The batch collector periodically gathers available frames from active cameras.
* A single TensorRT inference call is executed for the batch.
* Detection results are mapped back to the original camera index.
* The system prioritizes real-time responsiveness over processing every single frame.

This design improves GPU utilization and is suitable for edge AI systems, where GPU memory and computing resources are limited.

---

## 7. Multi-threaded Camera Capture

Each camera stream is handled in a separate thread.

This design prevents one unstable camera from blocking the entire system.
If one RTSP stream is disconnected or delayed, other camera streams can continue running.

The system keeps only the latest frame from each camera.
This is useful for real-time monitoring because displaying outdated frames would increase latency and reduce system reliability.

---

## 8. Hardware Constraint

The project was designed for NVIDIA Jetson Nano 4GB.

Because Jetson Nano has limited RAM and GPU resources, the system requires careful optimization.

Recommended performance mode:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Recommended camera configuration:

* Use sub-stream instead of main-stream
* Prefer VGA or low-resolution RTSP streams
* Avoid Full HD streams when running many cameras simultaneously
* Keep only the latest frame for each stream
* Use batched inference instead of per-camera inference

---

## 9. Performance Benchmark

| Item                   | Result                           |
| ---------------------- | -------------------------------- |
| Device                 | NVIDIA Jetson Nano 4GB           |
| Number of Cameras      | Up to 12 RTSP streams            |
| Display Layout         | 3x4 grid                         |
| AI Model               | YOLOv4-tiny                      |
| Inference Optimization | TensorRT FP16                    |
| Inference Strategy     | Batched Inference                |
| Video Input            | RTSP via GStreamer               |
| Target                 | Low-latency real-time monitoring |

Note: actual performance depends on camera resolution, RTSP bitrate, network condition, model size, and Jetson Nano power mode.

---

## 10. How to Run

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Configure RTSP camera links

Open `main12cam.py` and update the `RTSP_LINKS` list.

```python
RTSP_LINKS = [
    "rtsp://username:password@camera_ip_1/path",
    "rtsp://username:password@camera_ip_2/path",
    "rtsp://username:password@camera_ip_3/path",
]
```

For safety, real camera credentials are not included in this repository.

### 3. Run the system

```bash
python3 main12cam.py
```

Press `q` to exit.

---

## 11. Project Structure

```text
.
├── main12cam.py
├── requirements.txt
├── README.md
└── assets/
    └── demo_screenshot.png
```

Recommended future structure:

```text
.
├── src/
│   ├── camera_stream.py
│   ├── detector.py
│   ├── batch_inference.py
│   ├── monitor_view.py
│   └── main.py
├── docs/
│   ├── architecture.md
│   ├── performance.md
│   └── roadmap.md
├── assets/
│   └── demo_screenshot.png
├── requirements.txt
└── README.md
```

---

## 12. My Role

This was an individual project. I was responsible for:

* System design
* RTSP camera stream handling
* GStreamer pipeline construction
* Multi-threaded camera capture
* Latest-frame buffering strategy
* Batched inference design
* YOLOv4-tiny inference integration
* TensorRT-based model optimization
* Real-time multi-camera visualization
* Performance testing on Jetson Nano

---

## 13. Technical Challenges

### 1. GPU bottleneck with multiple cameras

Processing each camera independently caused too many small inference requests.
To solve this, I used batched inference to combine frames from multiple cameras into one inference call.

### 2. Latency accumulation

If old frames are stored and processed continuously, the displayed video becomes delayed.
To solve this, I used a latest-frame strategy and dropped old frames.

### 3. Limited Jetson Nano resources

Jetson Nano has limited RAM and GPU performance.
To reduce resource usage, I used TensorRT FP16, sub-stream RTSP input, GStreamer buffer control, and reduced display resolution.

### 4. Unstable RTSP connections

In real camera systems, some streams may disconnect or freeze.
To avoid stopping the entire application, each camera is handled in a separate thread.

---

## 14. Future Improvements

The next development direction is to extend this project into an AI-assisted RTSP stream quality monitoring dashboard.

Planned features:

* RTSP stream health monitoring
* FPS / latency / connection status visualization
* Packet loss and frame drop estimation
* Automatic alarm when a stream becomes unstable
* Web-based monitoring dashboard
* Stream quality scoring
* Failure prediction using historical stream metrics

---

## 15. Japanese Summary

このプロジェクトは、Jetson Nano上で複数のIPカメラからRTSP映像を取得し、GStreamerによる低遅延映像処理、YOLOv4-tinyとTensorRT FP16によるエッジAI推論、OpenCVによるリアルタイム可視化を行うマルチカメラ映像監視システムです。

複数カメラを同時に処理する際、各カメラごとに個別にAI推論を実行すると、GPUへの小さな推論リクエストが大量に発生し、処理効率が低下するという課題がありました。そこで本プロジェクトでは、各カメラから取得した最新フレームを一度バッチとしてまとめ、YOLOv4-tinyのTensorRTエンジンに対して一括で推論を実行するバッチ推論方式を導入しました。

また、古いフレームを処理し続けると遅延が蓄積するため、各カメラでは最新フレームのみを保持し、リアルタイム性を優先する設計にしました。これにより、Jetson Nanoのような限られたGPUリソース上でも、複数RTSPストリームのAI解析をより安定して行えるようにしました。

本プロジェクトでは、単なるAI検出ではなく、映像入力、ストリーム処理、推論最適化、例外処理、可視化までを一つのシステムとして設計することを重視しました。
