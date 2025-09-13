
#pragma once

#include "TfLiteEngine.h"
#include "Preprocess.h"
#include "Types.h"
#include "Log.h"

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace neptune {

/**
 * @class EmotionRecognizer
 * @brief Handles emotion recognition using a pre-trained TensorFlow Lite model.
 */
class EmotionRecognizer {
public:
    static std::unique_ptr<EmotionRecognizer> create(const std::string& modelPath,
                                                     const NeptuneConfig& config);

    /**
     * @brief Performs emotion recognition on a cropped face image (BGR cv::Mat).
     * @param faceImage The input image containing a single face.
     * @return An EmotionResult with the predicted emotion and confidence.
     */
    EmotionResult predictEmotion(const cv::Mat& faceImage);

private:
    EmotionRecognizer(const NeptuneConfig& config);
    bool init(const std::string& modelPath);

    // Helpers
    static std::vector<float> softmax(const std::vector<float>& logits);
    static Emotion indexToEmotion(int idx); // dataset label → enum

    std::unique_ptr<neptune::TfLiteEngine> engine_;
    int inputWidth_;
    int inputHeight_;
    float minConfidence_;
    int numClasses_ = -1; // inferred dynamically
};

} // namespace neptune
