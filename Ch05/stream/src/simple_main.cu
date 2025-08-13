// simple_main.cpp - Simple demo matching the slides exactly

#include <iostream>
#include <cuda_runtime.h>
#include "../include/image_pipeline.h"

int main() {
    // Create pipeline and image exactly as shown in slide
    ImagePipeline pipeline;
    pipeline.addGaussianBlur(1.5f);
    pipeline.addSobelEdgeDetection(); 
    pipeline.addResize(640, 480);
    
    // Create input and output images
    ImagePipeline::Image input_image(1920, 1080, 3);
    ImagePipeline::Image output_image(640, 480, 3);
    
    // Process
    pipeline.process(input_image, output_image);
    
    std::cout << "Pipeline executed successfully!" << std::endl;
    
    return 0;
}