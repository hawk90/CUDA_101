#ifndef PIPELINE_H
#define PIPELINE_H

#include <cuda_runtime.h>
#include <vector>
#include <functional>
#include <memory>

// Color space enum
enum class ColorSpace {
    RGB,
    YUV,
    HSV,
    GRAYSCALE
};

// Image structure
struct Image {
    unsigned char* data;
    int width, height, channels;
    size_t pitch;  // For 2D memory alignment
    cudaStream_t stream;
    
    Image(int w, int h, int c, cudaStream_t s = 0);
    ~Image();
    
    // Disable copy, enable move
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    
    size_t size() const { return width * height * channels; }
};

// Pipeline stage base class
class PipelineStage {
public:
    virtual ~PipelineStage() = default;
    virtual void process(Image& img) = 0;
    virtual std::string name() const = 0;
};

// Color conversion stage
class ColorConversionStage : public PipelineStage {
private:
    ColorSpace from_, to_;
    
public:
    ColorConversionStage(ColorSpace from, ColorSpace to);
    void process(Image& img) override;
    std::string name() const override { return "ColorConversion"; }
};

// Gaussian blur stage
class GaussianBlurStage : public PipelineStage {
private:
    float sigma_;
    float* kernel_;
    int kernel_size_;
    
public:
    GaussianBlurStage(float sigma);
    ~GaussianBlurStage();
    void process(Image& img) override;
    std::string name() const override { return "GaussianBlur"; }
};

// Sobel edge detection stage
class SobelEdgeStage : public PipelineStage {
public:
    void process(Image& img) override;
    std::string name() const override { return "SobelEdge"; }
};

// Resize stage
class ResizeStage : public PipelineStage {
private:
    int new_width_, new_height_;
    
public:
    ResizeStage(int w, int h);
    void process(Image& img) override;
    std::string name() const override { return "Resize"; }
};

// Histogram equalization stage
class HistogramEqualizationStage : public PipelineStage {
private:
    int* histogram_;
    int* cdf_;
    
public:
    HistogramEqualizationStage();
    ~HistogramEqualizationStage();
    void process(Image& img) override;
    std::string name() const override { return "HistogramEq"; }
};

// Main pipeline class
class ImagePipeline {
private:
    std::vector<std::unique_ptr<PipelineStage>> stages_;
    std::vector<cudaStream_t> streams_;
    int num_streams_;
    bool use_pinned_memory_;
    
    // Performance metrics
    std::vector<float> stage_times_;
    
public:
    ImagePipeline(int num_streams = 4, bool use_pinned = true);
    ~ImagePipeline();
    
    // Add processing stages
    void addStage(std::unique_ptr<PipelineStage> stage);
    
    // Specific stage helpers
    void addColorConversion(ColorSpace from, ColorSpace to);
    void addGaussianBlur(float sigma);
    void addSobelEdgeDetection();
    void addResize(int width, int height);
    void addHistogramEqualization();
    
    // Process single image
    void process(Image& input, Image& output);
    
    // Process batch of images (multi-stream)
    void processBatch(std::vector<Image>& inputs, 
                     std::vector<Image>& outputs);
    
    // Process video stream
    void processStream(const std::string& input_file,
                      const std::string& output_file);
    
    // Get performance metrics
    void printMetrics() const;
    float getTotalTime() const;
    std::vector<float> getStagesTimes() const { return stage_times_; }
};

// Multi-stream video processor
class VideoStreamProcessor {
private:
    ImagePipeline pipeline_;
    int num_streams_;
    
    // Buffers for multi-buffering
    std::vector<unsigned char*> host_buffers_;
    std::vector<unsigned char*> device_buffers_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    
public:
    VideoStreamProcessor(int num_streams = 4);
    ~VideoStreamProcessor();
    
    // Configure pipeline
    void configurePipeline(ImagePipeline& pipeline);
    
    // Process video file
    void processVideo(const std::string& input_path,
                     const std::string& output_path,
                     int target_fps = 60);
    
    // Real-time streaming
    void streamRealtime(int camera_id = 0);
    
    // Benchmark performance
    void benchmark(int width, int height, int num_frames);
};

#endif // PIPELINE_H