#!/bin/bash
set -e

IMAGE_NAME="hog_cuda_env"

echo "[1/3] Building Docker Image (Môi trường GCC 11 + CUDA)..."
# --- SỬA LỖI DNS: Thêm cờ '--network host' ---
# Giúp quá trình build dùng mạng trực tiếp của máy thật để tải thư viện apt-get
sudo podman build --network host -t $IMAGE_NAME .

echo "[2/3] Compiling Project inside Container..."
# Chạy container để biên dịch code
# Cũng thêm --network host cho chắc chắn
sudo podman run --rm --network host --device nvidia.com/gpu=all --security-opt=label=disable -v $(pwd):/app $IMAGE_NAME /bin/bash -c "
    echo '--- Inside Container ---'
    # Fix permission issues just in case
    git config --global --add safe.directory /app

    nvcc --version
    mkdir -p build
    cd build
    # Xóa sạch build cũ để tránh xung đột cache
    rm -rf *

    echo '-> Running CMake...'
    cmake ..

    echo '-> Compiling...'
    make -j\$(nproc)

    echo '--- Build Finished Successfully ---'
"

echo "[3/3] Done! Executable is in ./build/HOG_App"
echo "To run it using GPU, use:"
echo "sudo podman run --rm --network host --device nvidia.com/gpu=all -v \$(pwd):/app $IMAGE_NAME ./build/HOG_App ./assets/image.jpg 3"
