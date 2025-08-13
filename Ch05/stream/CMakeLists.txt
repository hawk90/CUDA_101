cmake_minimum_required(VERSION 3.18)
project(ImagePipeline LANGUAGES CXX CUDA)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directories
include_directories(include)

# Find packages
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_definitions(-DHAVE_OPENCV)
endif()

# CUDA kernels library
add_library(image_kernels STATIC
    src/kernels/color_space.cu
    src/kernels/filters.cu
    src/kernels/geometric.cu
    src/kernels/histogram.cu
)

# Pipeline library
add_library(pipeline_lib STATIC
    src/pipeline.cpp
    src/video_processor.cpp
)

# Main executable
add_executable(image_pipeline_demo
    src/main.cpp
)

target_link_libraries(image_pipeline_demo
    image_kernels
    pipeline_lib
)

if(OpenCV_FOUND)
    target_link_libraries(image_pipeline_demo ${OpenCV_LIBS})
endif()

# Benchmark executable
add_executable(pipeline_benchmark
    benchmarks/benchmark.cpp
)

target_link_libraries(pipeline_benchmark
    image_kernels
    pipeline_lib
)

# Tests
enable_testing()

add_executable(test_color_space
    tests/test_color_space.cu
)
target_link_libraries(test_color_space image_kernels)
add_test(NAME ColorSpaceTest COMMAND test_color_space)

add_executable(test_filters
    tests/test_filters.cu
)
target_link_libraries(test_filters image_kernels)
add_test(NAME FiltersTest COMMAND test_filters)

add_executable(test_pipeline
    tests/test_pipeline.cpp
)
target_link_libraries(test_pipeline pipeline_lib image_kernels)
add_test(NAME PipelineTest COMMAND test_pipeline)

# Installation
install(TARGETS image_pipeline_demo pipeline_benchmark
        RUNTIME DESTINATION bin)

install(DIRECTORY include/
        DESTINATION include)

# Package configuration
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)