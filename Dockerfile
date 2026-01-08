# Sử dụng môi trường chuẩn của NVIDIA (Ubuntu 22.04 + CUDA 12.3 + GCC 11)
# Đây là môi trường "vàng" tương thích hoàn hảo.
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Cập nhật và cài đặt các thư viện cần thiết
# Không dùng dnf mà dùng apt (vì là Ubuntu container)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Lệnh mặc định khi chạy container
CMD ["/bin/bash"]