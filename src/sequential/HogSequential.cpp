#include "../../include/HogSequential.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

// --- Clean Code: Tuning Constants ---
static constexpr float MAG_THRESHOLD = 0.1f;
static constexpr float ANGLE_SCALE = 180.0f;
static constexpr float VIS_SCALE = 0.3f;

HogSequential::HogSequential() : HogDetector() {
    // Use ANGLE_SCALE here
    binScale = static_cast<float>(BIN_COUNT) / ANGLE_SCALE;
}

void HogSequential::computeGradients(const Mat& img, Mat& mag, Mat& ang) {
    mag.create(img.size(), CV_32F);
    ang.create(img.size(), CV_32F);
    
    int rows = img.rows;
    int cols = img.cols;
    int cn = img.channels();

    for (int y = 1; y < rows - 1; y++) {
        const uchar* ptr = img.ptr<uchar>(y);
        const uchar* ptrPrev = img.ptr<uchar>(y - 1);
        const uchar* ptrNext = img.ptr<uchar>(y + 1);
        float* magPtr = mag.ptr<float>(y);
        float* angPtr = ang.ptr<float>(y);

        for (int x = 1; x < cols - 1; x++) {
            float maxGradSq = -1.0f;
            float bestAngle = 0.0f;

            if (cn == 1) {
                float dx = static_cast<float>(ptr[x + 1]) - static_cast<float>(ptr[x - 1]);
                float dy = static_cast<float>(ptrNext[x]) - static_cast<float>(ptrPrev[x]);
                maxGradSq = dx*dx + dy*dy;
                bestAngle = fastAtan2(dy, dx);
            } 
            else {
                for (int c = 0; c < 3; c++) {
                    float dx = static_cast<float>(ptr[(x + 1) * 3 + c]) - static_cast<float>(ptr[(x - 1) * 3 + c]);
                    float dy = static_cast<float>(ptrNext[x * 3 + c]) - static_cast<float>(ptrPrev[x * 3 + c]);
                    float gradSq = dx*dx + dy*dy;
                    
                    if (gradSq > maxGradSq) {
                        maxGradSq = gradSq;
                        bestAngle = fastAtan2(dy, dx); 
                    }
                }
            }
            magPtr[x] = std::sqrt(maxGradSq);
            angPtr[x] = bestAngle;
        }
    }
}

void HogSequential::computeCells(const Mat& mag, const Mat& ang, vector<float>& cellHistograms, Size& gridSize) {
    int cellsX = mag.cols / CELL_WIDTH;
    int cellsY = mag.rows / CELL_HEIGHT;
    gridSize = Size(cellsX, cellsY);
    
    cellHistograms.assign(cellsX * cellsY * BIN_COUNT, 0.0f);

    for (int cy = 0; cy < cellsY; cy++) {
        int startY = cy * CELL_HEIGHT;
        int endY = std::min(startY + CELL_HEIGHT, mag.rows);

        for (int y = startY; y < endY; y++) {
            const float* magPtr = mag.ptr<float>(y);
            const float* angPtr = ang.ptr<float>(y);
            int validWidth = cellsX * CELL_WIDTH;

            for (int x = 0; x < validWidth; x++) {
                float m = magPtr[x];
                // Use Constant
                if (m < MAG_THRESHOLD) continue;

                float a = angPtr[x];
                
                // --- FIXED: Use ANGLE_SCALE ---
                if (a >= ANGLE_SCALE) a -= ANGLE_SCALE;
                if (a < 0.0f) a += ANGLE_SCALE;

                float exactBin = a * binScale;
                int b0 = static_cast<int>(exactBin);
                
                if (b0 >= BIN_COUNT) b0 = 0;
                int b1 = b0 + 1;
                if (b1 >= BIN_COUNT) b1 = 0;

                float w1 = exactBin - b0;
                float w0 = 1.0f - w1;

                int cx = x / CELL_WIDTH;
                int cellIdx = (cy * cellsX + cx) * BIN_COUNT;
                
                cellHistograms[cellIdx + b0] += m * w0;
                cellHistograms[cellIdx + b1] += m * w1;
            }
        }
    }
}

Mat HogSequential::drawHOG(const vector<float>& cellHistograms, const Size& gridSize, const Mat& originalImg) {
    Mat visual;
    if (originalImg.channels() == 1) cvtColor(originalImg, visual, COLOR_GRAY2BGR);
    else visual = originalImg.clone();
    
    // Use Constant
    visual = visual * VIS_SCALE; 

    int cellsX = gridSize.width;
    int cellsY = gridSize.height;
    
    // Use Constant
    float radPerBin = (CV_PI / 180.0f) * (ANGLE_SCALE / BIN_COUNT);
    
    float maxVal = 0.0f;
    for (float v : cellHistograms) if (v > maxVal) maxVal = v;
    if (maxVal <= 0.0f) maxVal = 1.0f;

    for (int y = 0; y < cellsY; y++) {
        for (int x = 0; x < cellsX; x++) {
            int cellIdx = (y * cellsX + x) * BIN_COUNT;
            Point center(x * CELL_WIDTH + CELL_WIDTH / 2, y * CELL_HEIGHT + CELL_HEIGHT / 2);

            for (int b = 0; b < BIN_COUNT; b++) {
                float magnitude = cellHistograms[cellIdx + b];
                if (magnitude < maxVal * 0.05f) continue; 

                float strength = magnitude / maxVal;
                float lineLen = (CELL_WIDTH / 2) * strength * 2.5f; 
                float angle = b * radPerBin; 
                
                line(visual, 
                     Point(center.x - cos(angle)*lineLen, center.y - sin(angle)*lineLen), 
                     Point(center.x + cos(angle)*lineLen, center.y + sin(angle)*lineLen), 
                     Scalar(0, 255 * strength, 255), 1, LINE_AA);
            }
        }
    }
    return visual;
}

Mat HogSequential::computeHOG(const Mat& input, bool visualize) {
    computeGradients(input, mag, ang);
    Size gridSize;
    computeCells(mag, ang, cellHistograms, gridSize);
    if (visualize) return drawHOG(cellHistograms, gridSize, input);
    return Mat(); 
}
