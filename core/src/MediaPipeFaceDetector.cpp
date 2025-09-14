
#include "neptune/MediaPipeFaceDetector.h"
#include <cmath>
#include <algorithm>
#include <memory>

namespace neptune {

MediaPipeFaceDetector::MediaPipeFaceDetector() {}

MediaPipeFaceDetector::~MediaPipeFaceDetector() {}

bool MediaPipeFaceDetector::initialize() {
    const std::string modelPath = "../models/face_detection.tflite";
    
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model_) {
        Log::error("MediaPipeFaceDetector", "Failed to load model from: " + modelPath);
        return false;
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    
    if (!interpreter_) {
        Log::error("MediaPipeFaceDetector", "Failed to build interpreter");
        return false;
    }
    
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        Log::error("MediaPipeFaceDetector", "Failed to allocate tensors");
        return false;
    }
    
    GenerateAnchors();
    Log::info("MediaPipeFaceDetector", "Initialized successfully with " + 
              std::to_string(anchors_.size()) + " anchors");
    
    return true;
}

float MediaPipeFaceDetector::GetScale(int index, size_t total) {
    if (total == 1)
        return (minScale + maxScale) / 2.0f;
    return minScale + (maxScale - minScale) * (static_cast<float>(index)) / (static_cast<float>(total - 1));
}

void MediaPipeFaceDetector::GenerateAnchors() {
    anchors_.clear();
    int layerId = 0;
    
    while (layerId < 4) { // 4 strides
        std::vector<float> scales;
        int firstSameStrideLayer = layerId;

        while (layerId < 4 && strides[firstSameStrideLayer] == strides[layerId]) {
            float scale = GetScale(layerId, 4);
            float nextScale = GetScale(layerId + 1, 4);
            float interpolated = std::sqrt(scale * nextScale);

            scales.push_back(scale);
            scales.push_back(interpolated);
            layerId++;
        }

        int featureHeight = static_cast<int>(
            std::ceil(1.0f * inputHeight / strides[firstSameStrideLayer]));
        int featureWidth = static_cast<int>(
            std::ceil(1.0f * inputWidth / strides[firstSameStrideLayer]));

        for (int y = 0; y < featureHeight; y++) {
            float centerY = (y + offset) / featureHeight;
            for (int x = 0; x < featureWidth; x++) {
                float centerX = (x + offset) / featureWidth;
                for (const auto& scale : scales) {
                    NormalizedRect anchor;
                    anchor.x_center = centerX;
                    anchor.y_center = centerY;
                    anchor.width = scale;
                    anchor.height = scale;
                    anchors_.push_back(anchor);
                }
            }
        }
    }
}

FaceBox MediaPipeFaceDetector::DecodeBox(const float* rawBox, const NormalizedRect& anchor) {
    FaceBox detection;

    // Decode bounding box
    float centerX = rawBox[0];
    float centerY = rawBox[1];
    float width = rawBox[2];
    float height = rawBox[3];

    detection.x = centerX / xScale * anchor.width + anchor.x_center - (width / 2.f);
    detection.y = centerY / yScale * anchor.height + anchor.y_center - (height / 2.f);
    detection.width = width / wScale * anchor.width;
    detection.height = height / hScale * anchor.height;

    detection.confidence = SigmoidScore(rawBox[4]);

    // Decode 468 landmarks (x, y, z)
    detection.landmarks.clear();
    const int num_landmarks = 468;
    for (int i = 0; i < num_landmarks; i++) {
        int idx = 5 + i * 3; // x, y, z
        float lx = rawBox[idx] / xScale * anchor.width + anchor.x_center;
        float ly = rawBox[idx + 1] / yScale * anchor.height + anchor.y_center;
        // float lz = rawBox[idx + 2]; // optional: depth
        detection.landmarks.emplace_back(lx, ly);
    }

    return detection;
}


float MediaPipeFaceDetector::SigmoidScore(float rawScore) {
    float clippedScore = std::min(scoreClippingThresh, std::max(-scoreClippingThresh, rawScore));
    return 1.0f / (1.0f + std::exp(-clippedScore));
}

float MediaPipeFaceDetector::CalculateOverlap(const FaceBox& a, const FaceBox& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    
    if (x2 < x1 || y2 < y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float unionArea = a.width * a.height + b.width * b.height - intersection;
    
    return intersection / unionArea;
}

std::vector<FaceBox> MediaPipeFaceDetector::FilterBoxes(const std::vector<FaceBox>& input) {
    if (input.empty()) return {};
    
    std::vector<FaceBox> output;
    std::vector<FaceBox> remaining = input;
    
    // Sort by confidence
    std::sort(remaining.begin(), remaining.end(), 
        [](const FaceBox& a, const FaceBox& b) {
            return a.confidence > b.confidence;
        });
    
    while (!remaining.empty()) {
        FaceBox best = remaining[0];
        output.push_back(best);
        
        // Remove overlapping boxes
        std::vector<FaceBox> newRemaining;
        for (size_t i = 1; i < remaining.size(); i++) {
            if (CalculateOverlap(best, remaining[i]) <= minSuppressionThreshold) {
                newRemaining.push_back(remaining[i]);
            }
        }
        remaining = newRemaining;
    }
    
    return output;
}

std::vector<FaceBox> MediaPipeFaceDetector::detectFaces(const cv::Mat& image) {
    std::vector<FaceBox> results;
    
    if (image.empty()) {
        Log::warn("MediaPipeFaceDetector", "Empty input image");
        return results;
    }
    
    // Preprocessing - simple resize and normalization
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputWidth, inputHeight));
    
    cv::Mat floatImage;
    resized.convertTo(floatImage, CV_32F, 1.0 / 127.5, -1.0); // Scale to [-1, 1]
    
    // Copy to input tensor
    float* input = interpreter_->typed_input_tensor<float>(0);
    if (!input) {
        Log::error("MediaPipeFaceDetector", "Failed to get input tensor");
        return results;
    }
    
    // Convert BGR to RGB and copy data
    if (floatImage.channels() == 3) {
        cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);
    }
    
    // Copy data in correct order (HWC format)
    memcpy(input, floatImage.data, inputWidth * inputHeight * 3 * sizeof(float));
    
    // Run inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        Log::error("MediaPipeFaceDetector", "Inference failed");
        return results;
    }
    
    // Get outputs - adjust indices based on your model
    float* boxes = interpreter_->typed_output_tensor<float>(0);
    float* scores = interpreter_->typed_output_tensor<float>(1);
    
    if (!boxes || !scores) {
        Log::error("MediaPipeFaceDetector", "Failed to get output tensors");
        return results;
    }
    
    // Process detections
    const int num_boxes = static_cast<int>(anchors_.size());
    const int num_coordinates = 16; // 4 box coords + 1 score + 2 keypoints * 2 coords each
    
    for (int i = 0; i < num_boxes; i++) {
        if (i >= anchors_.size()) break; // Safety check
        
        float confidence = SigmoidScore(scores[i]);
        
        if (confidence >= minScoreThresh) {
            FaceBox faceBox = DecodeBox(boxes + i * num_coordinates, anchors_[i]);
            faceBox.confidence = confidence;
            results.push_back(faceBox);
        }
    }
    
    // Apply non-maximum suppression
    results = FilterBoxes(results);
    
    // Scale coordinates back to original image size
    float scaleX = static_cast<float>(image.cols) / inputWidth;
    float scaleY = static_cast<float>(image.rows) / inputHeight;
    
    for (auto& face : results) {
        face.x *= scaleX;
        face.y *= scaleY;
        face.width *= scaleX;
        face.height *= scaleY;
        
        for (auto& landmark : face.landmarks) {
            landmark.x *= scaleX;
            landmark.y *= scaleY;
        }
    }
    
    Log::info("MediaPipeFaceDetector", "Detected " + std::to_string(results.size()) + " faces");
    return results;
}

} // namespace neptune