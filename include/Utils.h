#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class HogDetector; 

struct BenchmarkStats {
    int frameId;
    int width;
    int height;
    double timeMs;
};

class Utils {
public:
    static const int WIN_WIDTH = 64;
    static const int WIN_HEIGHT = 128;
    static cv::VideoCapture openVideo(const std::string& source);
    static void saveTimesToCSV(const std::string& filename, const std::vector<BenchmarkStats>& stats);
    static void saveFrame(const cv::Mat& resultImage, int frameId);
    static void runBenchmarkTask(HogDetector* detector, const std::string& inputPath, const std::string& methodName, const std::string& outputFileName);
};
