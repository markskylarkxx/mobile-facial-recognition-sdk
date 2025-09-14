//
// File: NeptuneFacialSDK/core/src/NeptuneSDK.cpp
//
// This file implements the NeptuneSDK class, which ties together all
// the individual SDK components.
//

#include "neptune/NeptuneSDK.h"
#include "neptune/LivenessChecker.h"


namespace neptune {

    

NeptuneSDK::NeptuneSDK(const NeptuneConfig& config) : config_(config) {}

std::unique_ptr<NeptuneSDK> NeptuneSDK::create(const NeptuneConfig& config) {
    auto sdk = std::unique_ptr<NeptuneSDK>(new NeptuneSDK(config));
    if (!sdk->init()) {
        Log::error("NeptuneSDK", "Failed to initialize SDK");
        return nullptr;
    }
    return sdk;
}

bool NeptuneSDK::init() {
    faceDetector_ = FaceDetector::create(config_.faceDetectionModelPath, config_);
    emotionRecognizer_ = EmotionRecognizer::create(config_.emotionModelPath, config_);
    livenessChecker_ = std::make_unique<LivenessChecker>(config_);
    
    return faceDetector_ && emotionRecognizer_;

    // return faceDetector_ && emotionRecognizer_ && livenessChecker_;
}


std::vector<NeptuneResult> NeptuneSDK::processImage(const cv::Mat& image) {
    std::vector<NeptuneResult> results;

    auto faces = faceDetector_->detectFaces(image);
    for (const auto& face : faces) {
        cv::Rect roi(face.x, face.y, face.width, face.height);
        cv::Mat faceCrop = image(roi).clone();

        EmotionResult emotion = emotionRecognizer_->predictEmotion(faceCrop);
        LivenessResult live = livenessChecker_->check(face);
        NeptuneResult processed;
        processed.hasFace = true;
        processed.faceBox = face;
        processed.emotion = emotion;
        processed.liveness = live;

        results.push_back(processed);
    }

    return results;
}




} // namespace neptune
