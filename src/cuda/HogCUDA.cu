#include "../../include/HogCUDA.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// --- HELPER MACROS ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Constants for math optimization
#define PI_F 3.14159265359f
#define RAD_TO_DEG (180.0f / PI_F)

// --- DEVICE HELPERS ---

// Force Read-Only Cache load (__ldg) for higher bandwidth on scattered reads
__device__ __forceinline__ float get_pixel_val(const unsigned char* __restrict__ img, int idx) {
    return __ldg(&img[idx]);
}

__device__ void get_pixel_gradient(
    const unsigned char* __restrict__ img,
    int x, int y, int step,
    float& outMag, float& outAngle
) {
    // 3-Channel offsets
    int idx = y * step + x * 3;
    int next_x = idx + 3;
    int prev_x = idx - 3;
    int next_y = (y + 1) * step + x * 3;
    int prev_y = (y - 1) * step + x * 3;

    float maxGradSq = -1.0f;
    float bestDx = 0.0f;
    float bestDy = 0.0f;

    // Unroll the channel loop for speed
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float val_nx = get_pixel_val(img, next_x + c);
        float val_px = get_pixel_val(img, prev_x + c);
        float val_ny = get_pixel_val(img, next_y + c);
        float val_py = get_pixel_val(img, prev_y + c);

        float dx = val_nx - val_px;
        float dy = val_ny - val_py;
        float gradSq = dx * dx + dy * dy;

        if (gradSq > maxGradSq) {
            maxGradSq = gradSq;
            bestDx = dx;
            bestDy = dy;
        }
    }

    outMag = sqrtf(maxGradSq);
    
    // Fast atan2 and conversion
    float angle = atan2f(bestDy, bestDx) * RAD_TO_DEG;
    
    // Normalize angle to [0, 180)
    if (angle < 0) angle += 360.0f;
    if (angle >= 180.0f) angle -= 180.0f;
    
    outAngle = angle;
}

// --- KERNEL ---

__global__ void compute_hog_kernel(
    const unsigned char* __restrict__ img, 
    float* __restrict__ hist, 
    int rows, 
    int cols, 
    int step
) {
    // Access constants directly from class (Clean Code)
    // Note: This requires the compiler to see the constexpr definition
    constexpr int CW = HogDetector::CELL_WIDTH;
    constexpr int CH = HogDetector::CELL_HEIGHT;
    constexpr int BINS = HogDetector::BIN_COUNT;

    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;

    // Total cells grid
    int gridCellsX = cols / CW;
    
    // Boundary check (Cell level)
    if (cx * CW >= cols || cy * CH >= rows) return;

    // Registers for histogram accumulation
    float localHist[BINS] = {0.0f};

    int startY = cy * CH;
    int startX = cx * CW;

    // Loop over pixels in the cell
    // Note: We skip the outer 1-pixel border of the IMAGE to avoid boundary checks inside
    for (int dy = 0; dy < CH; dy++) {
        int y = startY + dy;
        // Global boundary check (Image level)
        if (y < 1 || y >= rows - 1) continue;

        for (int dx = 0; dx < CW; dx++) {
            int x = startX + dx;
            if (x < 1 || x >= cols - 1) continue;

            float mag, angle;
            get_pixel_gradient(img, x, y, step, mag, angle);

            if (mag < 0.1f) continue; // Threshold

            // Linear Interpolation for Binning
            float exactBin = angle * (BINS / 180.0f);
            int b0 = (int)exactBin;
            int b1 = (b0 + 1) % BINS; // cleaner modulo arithmetic
            if (b0 >= BINS) b0 = 0;   // safety

            float w1 = exactBin - b0;
            float w0 = 1.0f - w1;

            localHist[b0] += mag * w0;
            localHist[b1] += mag * w1;
        }
    }

    // Write to Global Memory
    // Linear index for the specific cell
    int cellIdx = (cy * gridCellsX + cx) * BINS;
    
    #pragma unroll
    for (int i = 0; i < BINS; i++) {
        hist[cellIdx + i] = localHist[i];
    }
}

// --- CLASS IMPLEMENTATION ---

namespace {
    // Helper to calculate Grid/Block sizes
    void getLaunchConfig(int w, int h, dim3& grid, dim3& block) {
        block = dim3(16, 16);
        int cellsX = w / HogDetector::CELL_WIDTH;
        int cellsY = h / HogDetector::CELL_HEIGHT;
        grid = dim3(
            (cellsX + block.x - 1) / block.x, 
            (cellsY + block.y - 1) / block.y
        );
    }
}

HogCUDA::HogCUDA() : HogDetector() {
    d_img = nullptr;
    d_hist = nullptr;
    // Check device early
    int count;
    cudaGetDeviceCount(&count);
    if(count == 0) std::cerr << "[Error] No CUDA Device!" << std::endl;
}

HogCUDA::~HogCUDA() {
    cleanup();
}

void HogCUDA::allocateBuffers(int width, int height) {
    if (width == currentWidth && height == currentHeight) return;
    cleanup();

    currentWidth = width;
    currentHeight = height;

    size_t imgBytes = width * height * 3 * sizeof(unsigned char);
    
    int cellsX = width / CELL_WIDTH;
    int cellsY = height / CELL_HEIGHT;
    size_t histBytes = cellsX * cellsY * BIN_COUNT * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_img, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_hist, histBytes));
    
    cellHistograms.resize(cellsX * cellsY * BIN_COUNT);
}

void HogCUDA::cleanup() {
    if (d_img) cudaFree(d_img);
    if (d_hist) cudaFree(d_hist);
    d_img = nullptr;
    d_hist = nullptr;
}

cv::Mat HogCUDA::computeHOG(const cv::Mat& input, bool visualize) {
    cv::Mat img;
    // Ensure data is continuous for simple pointer arithmetic
    if (input.isContinuous()) img = input;
    else img = input.clone();

    allocateBuffers(img.cols, img.rows);

    // 1. Async Upload
    CUDA_CHECK(cudaMemcpyAsync(d_img, img.data, img.total() * img.elemSize(), 
                               cudaMemcpyHostToDevice));

    // 2. Launch
    dim3 block, grid;
    getLaunchConfig(img.cols, img.rows, grid, block);

    compute_hog_kernel<<<grid, block>>>(
        d_img, 
        d_hist, 
        img.rows, 
        img.cols, 
        (int)img.step
    );

    CUDA_CHECK(cudaGetLastError());

    // 3. Blocking Download (Synchronizes implicitly)
    CUDA_CHECK(cudaMemcpy(cellHistograms.data(), d_hist, 
                          cellHistograms.size() * sizeof(float), 
                          cudaMemcpyDeviceToHost));

    if (visualize) return cv::Mat::zeros(img.size(), CV_8UC3); // Placeholder
    return cv::Mat();
}
