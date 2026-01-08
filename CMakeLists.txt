cmake_minimum_required(VERSION 3.10)
project(HOG_App LANGUAGES CXX CUDA) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Packages
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
# find_package(OpenCL REQUIRED) # Có thể bỏ qua nếu chỉ test CUDA

# Sources
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/cuda/*.cu")

# Includes
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_definitions(-DUSE_CUDA)

# Executable
add_executable(HOG_App ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(HOG_App 
    PRIVATE
    ${OpenCV_LIBS} 
    OpenMP::OpenMP_CXX 
    CUDA::cudart
)

if(NOT MSVC)
    target_compile_options(HOG_App PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>)
    # RTX 3050 = sm_86
    target_compile_options(HOG_App PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_86 --use_fast_math>)
endif()