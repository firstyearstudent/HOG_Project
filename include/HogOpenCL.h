#pragma once
#include "HogDetector.h"
#include <vector>

// Platform handling
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class HogOpenCL : public HogDetector {
private:
    // OpenCL Resources
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernelHog; // Single Fused Kernel

    // GPU Memory (Minimal set for Zero-Copy)
    cl_mem d_input;     // Input Image
    cl_mem d_hist;      // Output Histograms
    
    // State tracking
    int currentWidth = 0;
    int currentHeight = 0;
    std::vector<float> cellHistograms;

    // Internal Helpers
    void initOpenCL();
    void compileKernels();
    void allocateBuffers(int width, int height);
    void cleanup();

public:
    HogOpenCL();
    ~HogOpenCL();
    
    cv::Mat computeHOG(const cv::Mat& input, bool visualize) override;
};
