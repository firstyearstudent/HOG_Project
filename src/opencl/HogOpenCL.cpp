#include "../../include/HogOpenCL.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstring> 

using namespace cv;
using namespace std;

// --- EMBEDDED KERNEL SOURCE (Optimized Fusion) ---
// Note: We use raw string literal R"(...)" for clean multi-line C code.
static const char* KERNEL_SOURCE = R"(
    #define CELL_WIDTH 8
    #define CELL_HEIGHT 8
    #define BIN_COUNT 9
    #define PI 3.14159265359f
    #define MAG_THRESHOLD 0.1f

    __kernel void compute_hog_fused(
        __global const uchar* img,    
        __global float* hist,         
        int rows, 
        int cols,
        int step,                     
        float binScale
    ) {
        // Thread Mapping: 1 Thread = 1 Cell
        int cx = get_global_id(0);
        int cy = get_global_id(1);
        int cellsX = get_global_size(0);
        
        // Bounds Check
        if (cx * CELL_WIDTH >= cols || cy * CELL_HEIGHT >= rows) return;

        // Private Memory (Registers) - Fastest access possible
        float localHist[BIN_COUNT];
        for(int i=0; i<BIN_COUNT; i++) localHist[i] = 0.0f;

        int startY = cy * CELL_HEIGHT;
        int startX = cx * CELL_WIDTH;
        
        // Loop over the 8x8 pixels in this cell
        for (int dy = 0; dy < CELL_HEIGHT; dy++) {
            int y = startY + dy;
            if (y <= 0 || y >= rows - 1) continue; 
            
            for (int dx = 0; dx < CELL_WIDTH; dx++) {
                int x = startX + dx;
                if (x <= 0 || x >= cols - 1) continue;

                // --- Gradient Calculation ---
                int next_x = y * step + (x + 1) * 3;
                int prev_x = y * step + (x - 1) * 3;
                int next_y = (y + 1) * step + x * 3;
                int prev_y = (y - 1) * step + x * 3;

                float maxGradSq = -1.0f;
                float bestDx = 0.0f;
                float bestDy = 0.0f;

                // Unrolled Channel Loop (BGR)
                for (int c = 0; c < 3; c++) {
                    float dx = (float)img[next_x + c] - (float)img[prev_x + c];
                    float dy = (float)img[next_y + c] - (float)img[prev_y + c];
                    float gradSq = dx*dx + dy*dy;
                    if (gradSq > maxGradSq) {
                        maxGradSq = gradSq;
                        bestDx = dx;
                        bestDy = dy;
                    }
                }
                
                // --- Binning Logic ---
                float m = sqrt(maxGradSq);
                if (m < MAG_THRESHOLD) continue;

                // Fast angle calculation
                float angle = atan2(bestDy, bestDx) * (180.0f / PI);
                if (angle < 0) angle += 360.0f;
                if (angle >= 180.0f) angle -= 180.0f;
                
                float exactBin = angle * binScale;
                int b0 = (int)exactBin;
                if (b0 >= BIN_COUNT) b0 = 0;
                
                int b1 = b0 + 1;
                if (b1 >= BIN_COUNT) b1 = 0;
                
                float w1 = exactBin - b0;
                float w0 = 1.0f - w1;

                // Accumulate to registers
                localHist[b0] += m * w0;
                localHist[b1] += m * w1;
            }
        }

        // Write Final Result to VRAM
        int cellIdx = (cy * cellsX + cx) * BIN_COUNT;
        for (int i = 0; i < BIN_COUNT; i++) {
            hist[cellIdx + i] = localHist[i];
        }
    }
)";

// --- C++ HOST IMPLEMENTATION ---

#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        throw std::runtime_error(std::string("[OpenCL Error] ") + msg + " Code: " + std::to_string(err)); \
    }

HogOpenCL::HogOpenCL() : HogDetector() {
    initOpenCL();
    compileKernels();
}

HogOpenCL::~HogOpenCL() {
    cleanup();
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

void HogOpenCL::initOpenCL() {
    cl_int err;
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms == 0) throw std::runtime_error("No OpenCL platforms found");

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

    cl_device_id bestDevice = NULL;
    long long bestScore = -1; 

    // Robust Device Selector
    for (const auto& platform : platforms) {
        cl_uint numDevices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (numDevices == 0) continue;
        
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);

        for (const auto& dev : devices) {
            cl_device_type type;
            cl_uint units;
            char name[256];
            
            clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, NULL);
            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
            
            std::string sName(name);
            long long score = 0;

            // 1. Prefer GPU over CPU
            if (type & CL_DEVICE_TYPE_GPU) score += 10000;

            // 2. DISCRETE GPU PRIORITY
            if (sName.find("Radeon") != std::string::npos || 
                sName.find("AMD") != std::string::npos || 
                sName.find("NVIDIA") != std::string::npos ||
                sName.find("Pro") != std::string::npos) {
                score += 100000; 
            }

            score += units; 

            if (score > bestScore) {
                bestScore = score;
                bestDevice = dev;
            }
        }
    }

    if (!bestDevice) throw std::runtime_error("No OpenCL device found");

    char name[128];
    clGetDeviceInfo(bestDevice, CL_DEVICE_NAME, 128, name, NULL);
    std::cout << "[OpenCL] Selected Device: " << name << std::endl;

    context = clCreateContext(NULL, 1, &bestDevice, NULL, NULL, &err);
    CHECK_CL(err, "Create Context");
    queue = clCreateCommandQueue(context, bestDevice, 0, &err);
    CHECK_CL(err, "Create Queue");
}

void HogOpenCL::compileKernels() {
    cl_int err;
    size_t len = strlen(KERNEL_SOURCE);
    program = clCreateProgramWithSource(context, 1, &KERNEL_SOURCE, &len, &err);
    CHECK_CL(err, "Create Program");

    // OPTIMIZATION: "-cl-fast-relaxed-math" enables hardware native instructions
    err = clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    
    if (err != CL_SUCCESS) {
        size_t logSize;
        cl_device_id dev;
        clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(dev), &dev, NULL);
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL);
        std::cerr << "[OpenCL Build Error]: " << log.data() << std::endl;
        throw std::runtime_error("OpenCL Kernel Build Failed");
    }

    kernelHog = clCreateKernel(program, "compute_hog_fused", &err);
    CHECK_CL(err, "Create Kernel");
}

void HogOpenCL::allocateBuffers(int width, int height) {
    if (width == currentWidth && height == currentHeight) return;
    cleanup();

    currentWidth = width;
    currentHeight = height;
    
    size_t pixelBytes = width * height * 3;
    int cellsX = width / CELL_WIDTH;
    int cellsY = height / CELL_HEIGHT;
    size_t histBytes = cellsX * cellsY * BIN_COUNT * sizeof(float);

    cl_int err;
    d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, pixelBytes, NULL, &err);
    d_hist = clCreateBuffer(context, CL_MEM_WRITE_ONLY, histBytes, NULL, &err);
    CHECK_CL(err, "Buffer Allocation");
    
    cellHistograms.resize(cellsX * cellsY * BIN_COUNT);
}

void HogOpenCL::cleanup() {
    if (d_input) clReleaseMemObject(d_input);
    if (d_hist) clReleaseMemObject(d_hist);
    d_input = NULL;
    d_hist = NULL;
}

cv::Mat HogOpenCL::computeHOG(const cv::Mat& input, bool visualize) {
    cl_int err;
    Mat img = input.isContinuous() ? input : input.clone();
    
    allocateBuffers(img.cols, img.rows);

    // 1. Upload Image
    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, 
                               img.total() * img.elemSize(), img.data, 0, NULL, NULL);
    
    int cellsX = img.cols / CELL_WIDTH;
    int cellsY = img.rows / CELL_HEIGHT;
    size_t globalSize[2] = { (size_t)cellsX, (size_t)cellsY };
    
    int rows = img.rows;
    int cols = img.cols;
    int step = (int)img.step;
    float binScale = (float)BIN_COUNT / 180.0f;

    clSetKernelArg(kernelHog, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernelHog, 1, sizeof(cl_mem), &d_hist);
    clSetKernelArg(kernelHog, 2, sizeof(int), &rows);
    clSetKernelArg(kernelHog, 3, sizeof(int), &cols);
    clSetKernelArg(kernelHog, 4, sizeof(int), &step);
    clSetKernelArg(kernelHog, 5, sizeof(float), &binScale);

    // 2. Launch Kernel
    // Note: Local workgroup size is NULL (auto), which works best for this specific logic
    err = clEnqueueNDRangeKernel(queue, kernelHog, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_CL(err, "Kernel Execution");

    // 3. Read Results
    err = clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0, 
                              cellHistograms.size() * sizeof(float), 
                              cellHistograms.data(), 0, NULL, NULL);

    if (visualize) return Mat::zeros(img.size(), CV_8UC3);
    return Mat();
}
