# CMakeLists.txt for CNN Project
cmake_minimum_required(VERSION 3.18)
project(cnn_cuda LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")
file(GLOB_RECURSE HOST_SOURCES "src/*.cpp" "src/*.cu")
list(REMOVE_ITEM HOST_SOURCES ${KERNEL_SOURCES})

# Create kernel library
add_library(cnn_kernels STATIC ${KERNEL_SOURCES})
target_link_libraries(cnn_kernels 
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
)

# Main executable
add_executable(cnn_demo ${HOST_SOURCES})
target_link_libraries(cnn_demo 
    cnn_kernels
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
)

# Test executable
if(BUILD_TESTS)
    enable_testing()
    file(GLOB TEST_SOURCES "tests/*.cpp" "tests/*.cu")
    add_executable(cnn_tests ${TEST_SOURCES})
    target_link_libraries(cnn_tests 
        cnn_kernels
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
    )
    add_test(NAME CNNTests COMMAND cnn_tests)
endif()

# Install targets
install(TARGETS cnn_demo DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/cnn_cuda)