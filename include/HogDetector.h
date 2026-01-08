#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class HogDetector {
public:
    // --- Single Source of Truth ---
    // constexpr is implicitly 'inline' and available to device code 
    // when included in .cu files.
    static constexpr int CELL_WIDTH = 8;
    static constexpr int CELL_HEIGHT = 8;
    static constexpr int BIN_COUNT = 9;

    HogDetector() = default;
    virtual ~HogDetector() = default;

    virtual cv::Mat computeHOG(const cv::Mat& input, bool visualize = true) = 0;
    
    virtual long long getFeatureCount(const cv::Size& imgSize) const {
        //
        int cellsX = imgSize.width / CELL_WIDTH;
        int cellsY = imgSize.height / CELL_HEIGHT;
        if (cellsX <= 1 || cellsY <= 1) return 0;
        return (long long)(cellsX - 1) * (cellsY - 1) * 36; 
    }
};
