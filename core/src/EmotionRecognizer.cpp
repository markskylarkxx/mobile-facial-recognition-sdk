
#include "neptune/EmotionRecognizer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace neptune {

// // The seven emotion classes from your MobileNet model
const std::vector<std::string> EMOTION_LABELS = {
    "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"
};
EmotionRecognizer::EmotionRecognizer(const NeptuneConfig& config)
    : inputWidth_(0), inputHeight_(0), minConfidence_(config.minEmotionConfidence) {}

std::unique_ptr<EmotionRecognizer> EmotionRecognizer::create(const std::string& modelPath,
                                                             const NeptuneConfig& config) {
    auto recognizer = std::unique_ptr<EmotionRecognizer>(new EmotionRecognizer(config));
    if (!recognizer->init(modelPath)) {
        Log::error("EmotionRecognizer", "Failed to initialize with model: " + modelPath);
        return nullptr;
    }
    return recognizer;
}

bool EmotionRecognizer::init(const std::string& modelPath) {
    engine_ = std::make_unique<TfLiteEngine>();

    if (!engine_->loadModel(modelPath)) {
        Log::error("EmotionRecognizer", "Failed to load TFLite model: " + modelPath);
        return false;
    }

    inputWidth_  = engine_->inputWidth();
    inputHeight_ = engine_->inputHeight();

    Log::info("EmotionRecognizer", "Model expects input: " +
        std::to_string(inputWidth_) + "x" + std::to_string(inputHeight_));

    if (inputWidth_ == 0 || inputHeight_ == 0) {
        Log::error("EmotionRecognizer", "Engine failed to get valid input dimensions from the model.");
        return false;
    }

    Log::info("EmotionRecognizer", "Number of output tensors: " +
        std::to_string(engine_->getNumOutputs()));

    for (int i = 0; i < engine_->getNumOutputs(); ++i) {
        const auto shape = engine_->getOutputTensorShape(i);
        std::string s = "[";
        for (size_t j = 0; j < shape.size(); ++j) {
            s += std::to_string(shape[j]);
            if (j + 1 < shape.size()) s += ", ";
        }
        s += "]";
        Log::info("EmotionRecognizer", "Output " + std::to_string(i) + " shape: " + s);

        if (shape.size() == 2 && shape[1] > 0) {
            numClasses_ = static_cast<int>(shape[1]);
            if (numClasses_ != EMOTION_LABELS.size()) {
                Log::error("EmotionRecognizer", "Model output size (" + std::to_string(numClasses_) +
                            ") does not match expected labels size (" + std::to_string(EMOTION_LABELS.size()) + ").");
            }
        }
    }

    return true;
}

std::vector<float> EmotionRecognizer::softmax(const std::vector<float>& logits) {
    std::vector<float> out(logits.size());
    if (logits.empty()) return out;

    float maxv = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = std::exp(static_cast<double>(logits[i] - maxv));
        sum += out[i];
    }
    if (sum <= std::numeric_limits<double>::min()) {
        float u = 1.0f / static_cast<float>(logits.size());
        std::fill(out.begin(), out.end(), u);
        return out;
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = static_cast<float>(out[i] / sum);
    }
    return out;
}

// Maps index to enum
// Maps model output index to Emotion enum
Emotion EmotionRecognizer::indexToEmotion(int idx) {
    switch (idx) {
        case 0: return Emotion::ANGER;
        case 1: return Emotion::DISGUST;
        case 2: return Emotion::FEAR;
        case 3: return Emotion::HAPPINESS;
        case 4: return Emotion::SADNESS;
        case 5: return Emotion::SURPRISE;
        case 6: return Emotion::NEUTRAL;
        default: return Emotion::UNKNOWN;
    }
}




EmotionResult EmotionRecognizer::predictEmotion(const cv::Mat& faceImage) {
    EmotionResult result{Emotion::UNKNOWN, 0.0f};

    if (!engine_ || numClasses_ < 0) {
        Log::error("EmotionRecognizer", "Engine not initialized or number of classes not set.");
        return result;
    }
    if (faceImage.empty()) {
        Log::error("EmotionRecognizer", "Empty input image");
        return result;
    }

    // --- Preprocess ---
    cv::Mat resized = neptune::img::Preprocess::resize(faceImage, inputWidth_, inputHeight_);
    
    // --- ADD THIS LINE TO CONVERT FROM BGR to RGB ---
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    std::vector<float> inputTensor = neptune::img::Preprocess::normalize(resized);

    if (!engine_->setInputTensor(inputTensor)) {
        Log::error("EmotionRecognizer", "Failed to set input tensor");
        return result;
    }
    if (!engine_->invoke()) {
        Log::error("EmotionRecognizer", "Inference failed");
        return result;
    }

    // --- Post-processing (looks correct) ---
    auto output = engine_->getOutputTensor(0);
    if (output.empty() || output.size() != numClasses_) {
        Log::error("EmotionRecognizer", "Empty or unexpected size of output tensor.");
        return result;
    }

    std::vector<float> probs = softmax(output);

    int best = static_cast<int>(std::distance(probs.begin(),
                        std::max_element(probs.begin(), probs.end())));
    float conf = probs[best];

    if (conf < minConfidence_) {
        result.emotion = Emotion::UNKNOWN;
        result.confidence = conf;
        return result;
    }

    result.emotion = indexToEmotion(best);
    result.confidence = conf;

    // Debug log
    std::string dbg = "Probabilities: ";
    for (size_t i = 0; i < probs.size(); ++i) {
        dbg += EMOTION_LABELS[i] + "=" + std::to_string(probs[i]) + " ";
    }
    Log::info("EmotionRecognizer", dbg);

    return result;
}
} // namespace neptune









