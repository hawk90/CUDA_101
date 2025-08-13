# Simple Makefile for Image Processing Pipeline

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_75

# Targets
all: image_pipeline_demo

image_pipeline_demo: src/simple_main.cpp src/kernels/kernel_launchers.cu
	$(NVCC) $(NVCC_FLAGS) -o image_pipeline_demo \
		src/simple_main.cpp \
		src/kernels/kernel_launchers.cu

clean:
	rm -f image_pipeline_demo

run: image_pipeline_demo
	./image_pipeline_demo

.PHONY: all clean run