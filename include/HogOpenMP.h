#pragma once
#include "HogDetector.h"

class HogOpenMP : public HogDetector {
private:
    float binScale; 
    
    // Member buffers for memory reuse
    cv::Mat mag, ang;
    std::vector<float> cellHistograms;

    // Internal Helpers
    void computeGradients(const cv::Mat& img, cv::Mat& mag, cv::Mat& ang);
    void computeCells(const cv::Mat& mag, const cv::Mat& ang, std::vector<float>& cellHistograms, cv::Size& gridSize);
    
    // We can reuse the visualization logic, or implement a basic one
    cv::Mat drawHOG(const std::vector<float>& cellHistograms, const cv::Size& gridSize, const cv::Mat& originalImg);

public:
    HogOpenMP();
    cv::Mat computeHOG(const cv::Mat& input, bool visualize) override;
};
