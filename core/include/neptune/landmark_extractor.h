
#pragma once
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <memory>
#include <vector>
#include "neptune/Types.h"

class LandmarkExtractor {
public:
    LandmarkExtractor(const std::string& modelPath);
    ~LandmarkExtractor() = default;

    // Extract landmarks for a face ROI (faceRect is relative to full image)
    std::vector<neptune::Point> Process(const cv::Mat& image, const cv::Rect& faceRect);

private:
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;
    int inputWidth = 0;
    int inputHeight = 0;
};


