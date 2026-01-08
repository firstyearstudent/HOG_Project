# Hệ Thống Phát Hiện Vật Thể HOG

Dự án này là một triển khai hiệu năng cao của thuật toán trích xuất đặc trưng **Histogram of Oriented Gradients (HOG)** để phát hiện vật thể. Mục tiêu chính là so sánh hiệu suất giữa 4 chế độ thực thi khác nhau, từ đó minh chứng sức mạnh của lập trình song song và tăng tốc phần cứng.

## Các Chế Độ Hỗ Trợ

1. **Sequential (CPU):** Thuật toán chạy tuần tự trên một luồng đơn (Baseline - Mốc chuẩn).
2. **OpenMP (CPU):** Song song hóa đa luồng trên CPU sử dụng thư viện OpenMP.
3. **OpenCL (GPU):** Phiên bản tăng tốc phần cứng đa nền tảng (Hoạt động trên AMD, Intel, NVIDIA và Apple Silicon).
4. **CUDA (NVIDIA GPU):** Phiên bản tối ưu hóa chuyên sâu dành riêng cho GPU NVIDIA (Yêu cầu CUDA Toolkit).

---

## 1. Cấu Trúc Dự Án

Dưới đây là tổ chức thư mục của dự án:

```text
HOG_Project/
├── assets/                 # Chứa dữ liệu đầu vào (video.mp4, image.jpg)
├── build/                  # Thư mục chứa file thực thi sau khi biên dịch
├── include/                # Các file header (.h) định nghĩa lớp và hàm
│   ├── HOGDetector.h       # Interface chung cho các bộ phát hiện
│   ├── HogSequential.h     # Header cho thuật toán tuần tự
│   ├── HogOpenMP.h         # Header cho thuật toán OpenMP
│   ├── HogOpenCL.h         # Header cho thuật toán OpenCL
│   ├── HogCUDA.h           # Header cho thuật toán CUDA
│   └── Utils.h             # Các tiện ích xử lý ảnh/video, đo thời gian
├── src/                    # Mã nguồn chính (.cpp)
│   ├── main.cpp            # Điểm bắt đầu của chương trình (Entry point)
│   ├── Utils.cpp           # Cài đặt các hàm tiện ích
│   ├── HogSequential.cpp   # Cài đặt thuật toán tuần tự
│   ├── HogOpenMP.cpp       # Cài đặt thuật toán OpenMP
│   ├── HogOpenCL.cpp       # Cài đặt thuật toán OpenCL
│   └── cuda/               # Thư mục chứa mã nguồn CUDA
│       └── HogCUDA.cu      # Kernel CUDA (.cu) chạy trên GPU
├── results/                # Nơi lưu file CSV kết quả và biểu đồ phân tích
│   └── analyze.py          # Script Python để vẽ biểu đồ so sánh
├── CMakeLists.txt          # File cấu hình biên dịch CMake
├── Dockerfile              # Cấu hình môi trường chuẩn (Ubuntu 22.04 + GCC 11 + CUDA 12)
├── build_container.sh      # Script tự động build dự án qua Podman/Docker
└── README.md               # Tài liệu hướng dẫn sử dụng

```

---

## 2. Yêu Cầu Hệ Thống

Để chạy dự án này, bạn cần cài đặt:

### **Công cụ biên dịch & Container**

* **Docker** hoặc **Podman** (Khuyên dùng Podman trên Fedora).
* **NVIDIA Container Toolkit** (Bắt buộc để Container nhận diện được GPU RTX 3050).
* **CMake** (Version 3.10 trở lên).
* **OpenCV** (Version 4.x).

### **Công cụ phân tích (Python)**

Để tạo biểu đồ hiệu năng, bạn cần Python 3 và các thư viện sau:

```bash
pip install pandas matplotlib
```

---

## 3. Hướng Dẫn Biên Dịch (Automated Build)

1. **Cấp quyền thực thi cho script:**
```bash
chmod +x build_container.sh
```


2. **Chạy script biên dịch:**
```bash
./build_container.sh
```


*Script này sẽ tự động tải môi trường, cài đặt thư viện phụ thuộc và biên dịch mã nguồn. Nếu thành công, file thực thi `HOG_App` sẽ được tạo trong thư mục `build/`.*

---

## 4. Hướng Dẫn Chạy Ứng Dụng

Ứng dụng xử lý file video hoặc hình ảnh và xuất các chỉ số hiệu năng ra file CSV trong thư mục `results/`.

### Cú pháp chung

```bash
<Lệnh_Chạy> <Đường_dẫn_input> <Mã_Chế_độ>
```

### Bảng Mã Chế Độ (Mode IDs)

| ID | Chế độ | Mô tả |
| --- | --- | --- |
| **0** | **Sequential** | Chạy đơn luồng tiêu chuẩn trên CPU. |
| **1** | **OpenMP** | Chạy song song đa luồng trên CPU. |
| **2** | **OpenCL** | Tăng tốc GPU (Tự động chọn GPU rời nếu có). |
| **3** | **CUDA** | Tăng tốc GPU NVIDIA (Chỉ chạy được khi build có hỗ trợ CUDA). |

---

### Các Lệnh Chạy Mẫu (Run Examples)

Do ứng dụng được biên dịch trong Container, hãy sử dụng các lệnh dưới đây để chạy.

> **Lưu ý:** Thay thế `video.mp4` bằng tên file thực tế của bạn trong thư mục `assets/`.

#### 1. Chạy Mode 1: OpenMP (CPU Đa luồng)

Chế độ này chạy trên CPU, sử dụng tất cả các nhân có sẵn để xử lý song song.

```bash
sudo podman run --rm --network host \
  --security-opt=label=disable \
  -v $(pwd):/app \
  hog_cuda_env \
  ./build/HOG_App ./assets/video.mp4 1
```

#### 2. Chạy Mode 2: OpenCL (GPU Đa nền tảng)

Chế độ này sử dụng thư viện OpenCL chuẩn. Trong môi trường Container NVIDIA này, nó sẽ chạy trên GPU thông qua tầng tương thích của NVIDIA.

```bash
sudo podman run --rm --network host \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v $(pwd):/app \
  hog_cuda_env \
  ./build/HOG_App ./assets/video.mp4 2
```

#### 3. Chạy Mode 3: CUDA (NVIDIA GPU - Hiệu năng cao)

Chế độ này chạy trực tiếp kernel CUDA.

```bash
sudo podman run --rm --network host \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v $(pwd):/app \
  hog_cuda_env \
  ./build/HOG_App ./assets/video.mp4 3
```

---

### Giải thích các tham số lệnh:

* `sudo podman run --rm`: Chạy container và tự động xóa nó sau khi chạy xong (giữ sạch máy).
* `--network host`: Sử dụng mạng máy chủ (tránh lỗi DNS/kết nối).
* `--device nvidia.com/gpu=all`: **Quan trọng!** Cấp quyền cho Container truy cập vào Card đồ họa NVIDIA.
* `--security-opt=label=disable`: Tắt SELinux tạm thời cho container này để nó có quyền đọc/ghi file video.
* `-v $(pwd):/app`: Ánh xạ thư mục hiện tại vào thư mục `/app` (Container) để chương trình đọc được file `video.mp4` và ghi được file kết quả `.csv`.

## 5. Phân Tích & Đánh Giá (Benchmarking)

Ứng dụng tự động lưu log thực thi vào các file `.csv` trong thư mục `results/`.

Để trực quan hóa sự khác biệt về hiệu năng:

1. **Chạy lần lượt các chế độ cần so sánh** (Ví dụ: chạy Mode 0, Mode 1 và Mode 3).
2. **Tạo biểu đồ:**
```bash
python3 results/analyze.py
```


3. **Xem kết quả:**
Mở thư mục `results/` để xem các biểu đồ đã tạo:
* `benchmark_fps.png`: So sánh tốc độ khung hình trung bình (Average FPS).
* `benchmark_timeline.png`: Phân tích độ ổn định (thời gian xử lý từng frame).
* `benchmark_cumulative.png`: Tổng thời gian trôi qua.

---

## 6. Khắc Phục Sự Cố (Troubleshooting)

* **Lỗi "Permission denied" khi chạy:**
    * Nguyên nhân: SELinux chặn Container đọc file.
    * Khắc phục: Đảm bảo lệnh `podman run` có tham số `--security-opt=label=disable`.
    * Cấp quyền chạy cho file thực thi: `sudo chmod +x build/HOG_App`.


* **Lỗi kết nối mạng khi build (`Temporary failure resolving...`):**
    * Nguyên nhân: Lỗi DNS trong Container.
    * Khắc phục: Script `build_container.sh` đã được cập nhật cờ `--network host`. Hãy đảm bảo bạn đang dùng phiên bản mới nhất của script.


* **CUDA chạy chậm hơn OpenCL?**
    * Nguyên nhân: Thời gian khởi tạo Driver (Warm-up) hoặc cấp phát bộ nhớ liên tục.
    * Khắc phục: Thử chạy với video dài hơn để bỏ qua chi phí khởi tạo ban đầu. Code đã được tối ưu để tránh cấp phát bộ nhớ (`cudaMalloc`) bên trong vòng lặp xử lý.


* **"OpenCL Error: No GPU found"**
    * Đảm bảo bạn đã cài đặt driver đồ họa mới nhất.
    * Trên Linux, cài đặt gói `clinfo` để kiểm tra khả năng nhận diện OpenCL.
