
#pragma once

#include "neptune/TfLiteEngine.h"
#include "Preprocess.h"
#include "Types.h"
#include "Log.h"

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace neptune {

/**
 * FaceDetector - detects faces using a TFLite model (supports MediaPipe 2-output SSD-style models).
 */
class FaceDetector {
public:
    // Creates a new FaceDetector instance.
    static std::unique_ptr<FaceDetector> create(const std::string& modelPath, const NeptuneConfig& config);

    // Perform detection on an OpenCV Mat (BGR). Returns FaceBox in original image coordinates.
    std::vector<FaceBox> detectFaces(const cv::Mat& image);

private:
    FaceDetector(const NeptuneConfig& config);
    bool init(const std::string& modelPath);

    // Legacy parsers (kept for compatibility)
    void parseMediaPipeFormat(const std::vector<float>& output, const cv::Mat& image, std::vector<FaceBox>& results);
    void parseSSDFormat(const cv::Mat& image, std::vector<FaceBox>& results);
    void parsePackedFormat(const std::vector<float>& output, const cv::Mat& image, std::vector<FaceBox>& results);
    void parseUnknownFormat(const std::vector<float>& output, const cv::Mat& image, std::vector<FaceBox>& results);

    // MediaPipe 2-output parser (boxes+keypoints, scores)
    void parseMediaPipe2OutputFormat(const std::vector<float>& boxes_and_keypoints,
                                     const std::vector<float>& scores,
                                     const cv::Mat& image,
                                     std::vector<FaceBox>& results);

    // Anchor type used for decoding SSD outputs (normalized coordinates)
    struct Anchor {
        float x_center; // normalized [0..1]
        float y_center; // normalized [0..1]
        float w;        // normalized [0..1]
        float h;        // normalized [0..1]
    };

    // Generate anchors used by MediaPipe face detectors
    std::vector<Anchor> generateAnchors(int input_width,
                                        int input_height,
                                        const std::vector<int>& strides,
                                        float min_scale,
                                        float max_scale,
                                        float anchor_offset_x = 0.5f,
                                        float anchor_offset_y = 0.5f);

    // The TensorFlow Lite engine used for running the face detection model.
    std::unique_ptr<neptune::TfLiteEngine> engine_;

    // Input tensor dims & thresholds
    int inputWidth_;
    int inputHeight_;
    float minConfidence_;

    // Cached anchors for the active model (filled at init or on-demand).
    std::vector<Anchor> anchors_;
};

} // namespace neptune







