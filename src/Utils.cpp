#include "../include/Utils.h"
#include "../include/HogDetector.h" 
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <filesystem>
#include <numeric>
#include <chrono>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// ==========================================
// CONFIGURATION
// ==========================================
// Set to TRUE to debug/see images. Set to FALSE for maximum speed benchmarking.
static constexpr bool SAVE_OUTPUT = true; 

// Minimum frames to process. 
// If video is short, we loop it to ensure CPU/GPU "warms up" and we get stable stats.
static constexpr int MIN_BENCHMARK_FRAMES = 20000; 
// ==========================================

// [DELETED] static long long calculateFeatureCount... (Redundant)

VideoCapture Utils::openVideo(const string& source) {
    VideoCapture cap;
    if (isdigit(source[0]) && source.size() == 1) cap.open(stoi(source));
    else cap.open(source);
    if (!cap.isOpened()) cerr << "[Error] Cannot open video source: " << source << endl;
    return cap;
}

void Utils::saveTimesToCSV(const string& filename, const vector<BenchmarkStats>& stats) {
    string outputDir = "../results/";
    if (!fs::exists(outputDir)) fs::create_directories(outputDir);

    ofstream file(outputDir + filename);
    if (!file.is_open()) return;

    file << "Frame,Width,Height,Time_ms\n";
    for (const auto& s : stats) {
        file << s.frameId << "," << s.width << "," << s.height << "," << s.timeMs << "\n";
    }
    cout << "[Saved] Stats to " << outputDir << filename << endl;
}

void Utils::saveFrame(const Mat& resultImage, int frameId) {
    if (resultImage.empty()) return;
    
    string outputDir = "../results/output_frames/";
    if (!fs::exists(outputDir)) fs::create_directories(outputDir);

    stringstream ss;
    ss << outputDir << "viz_" << setfill('0') << setw(4) << frameId << ".jpg";
    
    if (!imwrite(ss.str(), resultImage)) {
        cerr << "[Error] Could not save frame to " << ss.str() << endl;
    }
}

// --- BENCHMARK LOGIC ---

static void processFrameInternal(HogDetector* detector, Mat& img, int id, vector<BenchmarkStats>& stats) {
    if (img.empty()) return;

    // 1. Start Timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // 2. Run Algorithm
    Mat visual = detector->computeHOG(img, SAVE_OUTPUT);
    
    // 3. Stop Timer
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    BenchmarkStats s;
    s.frameId = id;
    s.width = img.cols;
    s.height = img.rows;
    s.timeMs = ms;
    stats.push_back(s);

    // 4. Logging
    // CLEAN CODE FIX: Use the detector's method instead of duplicating math here.
    long long featureCount = detector->getFeatureCount(img.size());

    if (SAVE_OUTPUT) {
        cout << "ID " << id << " [" << img.cols << "x" << img.rows << "]: " 
             << ms << " ms (" << featureCount << " features) [Saved]" << endl;
        Utils::saveFrame(visual, id);
    } else {
        if (id % 100 == 0 || id == 0) {
            cout << "ID " << id << " [" << img.cols << "x" << img.rows << "]: " 
                 << ms << " ms (" << featureCount << " features)" << endl;
        }
    }
}

void Utils::runBenchmarkTask(HogDetector* detector, const string& inputPath, const string& methodName, const string& outputFileName) {
    cout << "\n=== Running Benchmark: " << methodName << " ===" << endl;
    
    if (SAVE_OUTPUT) cout << "[WARNING] SAVE_OUTPUT is ON. Performance will be lower due to I/O." << endl;
    else cout << "[INFO] SAVE_OUTPUT is OFF. Running pure algorithm speed test." << endl;
    
    vector<BenchmarkStats> stats;
    vector<string> imageFiles;
    VideoCapture cap;
    bool isVideo = false;

    if (inputPath == "0" || (isdigit(inputPath[0]) && inputPath.size() == 1)) {
        isVideo = true;
        cap = Utils::openVideo(inputPath);
    } 
    else if (fs::is_directory(inputPath)) {
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                imageFiles.push_back(entry.path().string());
            }
        }
        sort(imageFiles.begin(), imageFiles.end());
        cout << "[Mode] Directory: " << imageFiles.size() << " images found." << endl;
    } 
    else if (fs::exists(inputPath)) {
        string ext = fs::path(inputPath).extension().string();
        if (ext == ".mp4" || ext == ".avi" || ext == ".mov") {
            isVideo = true;
            cap = Utils::openVideo(inputPath);
            cout << "[Mode] Video File" << endl;
        } else {
            imageFiles.push_back(inputPath);
            cout << "[Mode] Single Image" << endl;
        }
    } 
    else {
        cerr << "[Error] Invalid path: " << inputPath << endl;
        return;
    }

    if (isVideo) {
        Mat frame;
        int frameIdx = 0;
        
        while (frameIdx < MIN_BENCHMARK_FRAMES) {
            cap >> frame;
            if (frame.empty()) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }
            processFrameInternal(detector, frame, frameIdx++, stats);
        }
        cout << "[Done] Processed " << frameIdx << " frames (Virtual Loop)." << endl;
    } else {
        int imgIdx = 0;
        for (const auto& file : imageFiles) {
            Mat img = imread(file);
            processFrameInternal(detector, img, imgIdx++, stats);
        }
        if (imageFiles.empty() && !isVideo) {
             Mat img = imread(inputPath);
             if (!img.empty()) {
                 cout << "[Info] Looping single image for benchmark stability..." << endl;
                 for(int i=0; i < MIN_BENCHMARK_FRAMES; i++) {
                     processFrameInternal(detector, img, i, stats);
                 }
             }
        }
    }

    if (!stats.empty()) {
        Utils::saveTimesToCSV(outputFileName, stats);
    }
}
