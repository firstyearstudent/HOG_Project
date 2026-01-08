#pragma once
#include "HogDetector.h"
#include <vector>

// Forward declaration to avoid including cuda_runtime.h in the header
// (Keeps compile times fast for non-CUDA files)
struct float3; 

class HogCUDA : public HogDetector {
private:
    // Device (GPU) Pointers
    unsigned char* d_img;
    float* d_hist;
    
    // Host (CPU) Mirror
    std::vector<float> cellHistograms;

    int currentWidth = 0;
    int currentHeight = 0;

    void allocateBuffers(int width, int height);
    void cleanup();

public:
    HogCUDA();
    ~HogCUDA();

    cv::Mat computeHOG(const cv::Mat& input, bool visualize) override;
};
