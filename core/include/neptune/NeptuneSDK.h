//
// File: NeptuneFacialSDK/core/include/neptune/NeptuneSDK.h
//
// This file declares the NeptuneSDK class, which is the main public façade
// for the SDK.
//

#pragma once

#include "Types.h"
#include "FaceDetector.h"
#include "EmotionRecognizer.h"
 #include "LivenessChecker.h"
#include "Preprocess.h"
#include "Log.h"

#include <memory>
#include <string>
#include <vector>

namespace neptune {

/**
 * @class NeptuneSDK
 * @brief The main façade for the Neptune Facial SDK.
 *
 * This class provides a high-level, unified interface for all SDK capabilities,
 * including face detection, emotion recognition, and liveness checking.
 */
class NeptuneSDK {
public:
    /**
     * @brief Creates a new NeptuneSDK instance.
     * @param config The configuration settings for the SDK.
     * @return A unique pointer to the created SDK instance, or nullptr on failure.
     */
    static std::unique_ptr<NeptuneSDK> create(const NeptuneConfig& config);

    /**
     * @brief Processes a single image to detect faces, recognize emotions, and check liveness.
     * @param image The input image in OpenCV Mat format.
     * @return A vector of ProcessedFace objects, one for each face found.
     */
    std::vector<NeptuneResult> processImage(const cv::Mat& image);

private:
    // Private constructor to enforce creation via the static `create` method.
    NeptuneSDK(const NeptuneConfig& config);

    // Private initialization method.
    bool init();

    // The individual SDK components.
    std::unique_ptr<FaceDetector> faceDetector_;
    std::unique_ptr<EmotionRecognizer> emotionRecognizer_;
    std::unique_ptr<LivenessChecker> livenessChecker_;   // enable this


    // SDK configuration.
    NeptuneConfig config_;
};

} // namespace neptune
