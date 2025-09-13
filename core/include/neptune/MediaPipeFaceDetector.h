// // File: core/include/neptune/MediaPipeFaceDetector.h
// #pragma once

// #include "neptune/Types.h"
// #include <opencv2/opencv.hpp>
// #include <vector>
// #include <memory>

// namespace neptune {

// class MediaPipeFaceDetector {
// public:
//     MediaPipeFaceDetector();
//     ~MediaPipeFaceDetector();
    
//     bool initialize();
//     std::vector<FaceBox> detectFaces(const cv::Mat& image);
    
//     bool isInitialized() const { return initialized_; }
    
// private:
//     bool initialized_ = false;
//     // Implementation details...
// };

// } // namespace neptune





///////////////............................. use this!
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include "neptune/Types.h"
#include "neptune/Log.h"

namespace neptune {



class MediaPipeFaceDetector {
public:
    MediaPipeFaceDetector();
    ~MediaPipeFaceDetector();
    
    bool initialize();
    std::vector<FaceBox> detectFaces(const cv::Mat& image);
    
private:
    float GetScale(int index, size_t total);
    void GenerateAnchors();
    FaceBox DecodeBox(const float* rawBox, const NormalizedRect& anchor);
    float SigmoidScore(float rawScore);
    std::vector<FaceBox> FilterBoxes(const std::vector<FaceBox>& input);
    float CalculateOverlap(const FaceBox& a, const FaceBox& b);
    
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::vector<NormalizedRect> anchors_;
    
    // Constants
    static constexpr float minScale = 0.1484375f;
    static constexpr float maxScale = 0.75f;
    static constexpr int strides[4] = { 8, 16, 16, 16 };
    static constexpr float offset = 0.5f;
    static constexpr float xScale = 128.0f;
    static constexpr float yScale = 128.0f;
    static constexpr float hScale = 128.0f;
    static constexpr float wScale = 128.0f;
    static constexpr float minScoreThresh = 0.5f;
    static constexpr float scoreClippingThresh = 100.0f;
    static constexpr float minSuppressionThreshold = 0.3f;
    static constexpr int inputWidth = 128;
    static constexpr int inputHeight = 128;
};

} // namespace neptune