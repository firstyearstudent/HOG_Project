#pragma once
#include "HogDetector.h"

class HogSequential : public HogDetector {
private:
    // Optimization: Multiply by scale instead of dividing by step
    float binScale; 

    // --- Memory Reuse ---
    cv::Mat mag, ang;
    std::vector<float> cellHistograms;
    // --------------------

    void computeGradients(const cv::Mat& img, cv::Mat& mag, cv::Mat& ang);
    void computeCells(const cv::Mat& mag, const cv::Mat& ang, std::vector<float>& cellHistograms, cv::Size& gridSize);
    cv::Mat drawHOG(const std::vector<float>& cellHistograms, const cv::Size& gridSize, const cv::Mat& originalImg);

public:
    HogSequential();
    cv::Mat computeHOG(const cv::Mat& input, bool visualize) override;
};

