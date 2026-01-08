#include <iostream>
#include <string>
#include "../include/HogSequential.h"
#include "../include/HogOpenMP.h"
#include "../include/HogOpenCL.h"
#include "../include/Utils.h"

// Only include CUDA header if CMake found the toolkit
#ifdef USE_CUDA
#include "../include/HogCUDA.h"
#endif

int main(int argc, char** argv) {
    std::string input = (argc > 1) ? argv[1] : "../assets/image.jpg";
    int mode = (argc > 2) ? std::stoi(argv[2]) : 0;

    HogDetector* detector = nullptr;
    std::string name;
    std::string csvName;

    switch (mode) {
        case 1:
            std::cout << "[Mode] OpenMP Parallel CPU" << std::endl;
            detector = new HogOpenMP();
            name = "OpenMP CPU";
            csvName = "OpenMP.csv";
            break;
        case 2:
            std::cout << "[Mode] OpenCL GPU" << std::endl;
            detector = new HogOpenCL();
            name = "OpenCL GPU";
            csvName = "OpenCL.csv";
            break;
        case 3:
            #ifdef USE_CUDA
                std::cout << "[Mode] CUDA NVIDIA GPU" << std::endl;
                detector = new HogCUDA();
                name = "CUDA GPU";
                csvName = "CUDA.csv";
            #else
                std::cerr << "[Error] This executable was compiled without CUDA support." << std::endl;
                return 1;
            #endif
            break;
        default:
            std::cout << "[Mode] Sequential CPU" << std::endl;
            detector = new HogSequential();
            name = "Sequential CPU";
            csvName = "Sequential.csv";
            break;
    }
    
    if (detector) {
        Utils::runBenchmarkTask(detector, input, name, csvName);
        delete detector;
    }

    return 0;
}
