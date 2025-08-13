# CUDA CNN Project Makefile

# Compiler and flags
NVCC = nvcc
CXX = g++
CUDA_FLAGS = -O3 -arch=sm_75 -use_fast_math
CXX_FLAGS = -O3 -std=c++17
LDFLAGS = -lcudnn -lcublas

# Directories
SRC_DIR = src
KERNEL_DIR = $(SRC_DIR)/kernels
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
KERNEL_SOURCES = $(wildcard $(KERNEL_DIR)/*.cu)
HOST_SOURCES = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp)

# Object files
KERNEL_OBJECTS = $(patsubst $(KERNEL_DIR)/%.cu,$(BUILD_DIR)/%.o,$(KERNEL_SOURCES))
HOST_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(HOST_SOURCES:.cpp=.o))

# Target executable
TARGET = $(BIN_DIR)/cnn_demo

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Link executable
$(TARGET): $(KERNEL_OBJECTS) $(HOST_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(LDFLAGS)

# Compile CUDA kernels
$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu
	$(NVCC) $(CUDA_FLAGS) -I$(INC_DIR) -c $< -o $@

# Compile host code
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDA_FLAGS) -I$(INC_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -I$(INC_DIR) -I/usr/local/cuda/include -c $< -o $@

# Run the demo
run: $(TARGET)
	./$(TARGET)

# Benchmark
benchmark: $(TARGET)
	./$(TARGET) --benchmark

# Profile with nvprof
profile: $(TARGET)
	nvprof --print-gpu-trace ./$(TARGET)

# Profile with Nsight Compute
nsight: $(TARGET)
	ncu --set full ./$(TARGET)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all directories run benchmark profile nsight clean