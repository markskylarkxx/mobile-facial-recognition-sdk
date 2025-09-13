
#include "neptune/FaceDetector.h"
#include "neptune/Log.h"
#include "neptune/Preprocess.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace neptune {

// Sigmoid helper
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ------------------- Constructor / create / init -------------------
FaceDetector::FaceDetector(const NeptuneConfig& config)
    : inputWidth_(0), inputHeight_(0), minConfidence_(config.minFaceDetectionConfidence) {}

std::unique_ptr<FaceDetector> FaceDetector::create(const std::string& modelPath, const NeptuneConfig& config) {
    auto detector = std::unique_ptr<FaceDetector>(new FaceDetector(config));
    if (!detector->init(modelPath)) {
        Log::error("FaceDetector", "Failed to initialize with model: " + modelPath);
        return nullptr;
    }
    return detector;
}

bool FaceDetector::init(const std::string& modelPath) {
    engine_ = std::make_unique<neptune::TfLiteEngine>();
    if (!engine_->loadModel(modelPath)) {
        Log::error("FaceDetector", "Failed to load TFLite model: " + modelPath);
        return false;
    }

    inputWidth_ = engine_->inputWidth();
    inputHeight_ = engine_->inputHeight();

    Log::info("FaceDetector", "Model expects input: " +
              std::to_string(inputWidth_) + "x" + std::to_string(inputHeight_));

    anchors_.clear();
    return true;
}

// ------------------- Anchor generator -------------------
std::vector<FaceDetector::Anchor> FaceDetector::generateAnchors(int input_width,
                                                                int input_height,
                                                                const std::vector<int>& strides,
                                                                float min_scale,
                                                                float max_scale,
                                                                float anchor_offset_x,
                                                                float anchor_offset_y) {
    std::vector<Anchor> anchors;
    int num_layers = static_cast<int>(strides.size());
    if (num_layers <= 0) return anchors;

    std::vector<float> scales(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        scales[i] = (num_layers == 1) ? 0.5f * (min_scale + max_scale)
                                      : min_scale + (max_scale - min_scale) * i / (num_layers - 1);
    }

    for (int layer = 0; layer < num_layers; ++layer) {
        int stride = strides[layer];
        int fm_w = static_cast<int>(std::ceil(static_cast<float>(input_width) / stride));
        int fm_h = static_cast<int>(std::ceil(static_cast<float>(input_height) / stride));

        float scale = scales[layer];
        float scale_next = (layer == num_layers - 1) ? 1.0f : scales[layer + 1];
        float scale_geom = std::sqrt(scale * scale_next);

        for (int y = 0; y < fm_h; ++y) {
            for (int x = 0; x < fm_w; ++x) {
                float x_center = (x + anchor_offset_x) / fm_w;
                float y_center = (y + anchor_offset_y) / fm_h;

                anchors.push_back(Anchor{x_center, y_center, scale, scale});
                anchors.push_back(Anchor{x_center, y_center, scale_geom, scale_geom});
            }
        }
    }
    return anchors;
}

// ------------------- Non-Max Suppression -------------------
static float iouBox(const FaceBox& a, const FaceBox& b) {
    int inter_w = std::max(0, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
    int inter_h = std::max(0, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
    float inter = static_cast<float>(inter_w) * inter_h;
    float areaA = static_cast<float>(a.width) * a.height;
    float areaB = static_cast<float>(b.width) * b.height;
    return inter / (areaA + areaB - inter + 1e-6f);
}

static std::vector<FaceBox> nonMaxSuppression(std::vector<FaceBox>& boxes, float iou_threshold, int top_k = 5) {
    std::sort(boxes.begin(), boxes.end(), [](const FaceBox& a, const FaceBox& b){ return a.confidence > b.confidence; });
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<FaceBox> out;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        out.push_back(boxes[i]);
        if (static_cast<int>(out.size()) >= top_k) break;

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (iouBox(boxes[i], boxes[j]) > iou_threshold) suppressed[j] = true;
        }
    }
    return out;
}

// ------------------- MediaPipe 2-output parser -------------------
void FaceDetector::parseMediaPipe2OutputFormat(const std::vector<float>& boxes_and_keypoints,
                                               const std::vector<float>& scores,
                                               const cv::Mat& image,
                                               std::vector<FaceBox>& results) {
    if (scores.empty() || boxes_and_keypoints.empty()) return;
    int N = static_cast<int>(scores.size());

    if (anchors_.empty() || static_cast<int>(anchors_.size()) != N) {
        anchors_ = generateAnchors(inputWidth_, inputHeight_, {8,16,16,16}, 0.1484375f, 0.75f, 0.5f, 0.5f);
    }

    float ratio = std::min(static_cast<float>(inputWidth_) / image.cols,
                           static_cast<float>(inputHeight_) / image.rows);
    int pad_x = static_cast<int>((inputWidth_ - image.cols * ratio) * 0.5f);
    int pad_y = static_cast<int>((inputHeight_ - image.rows * ratio) * 0.5f);

    float x_scale = static_cast<float>(inputWidth_);
    float y_scale = static_cast<float>(inputHeight_);

    std::vector<FaceBox> decoded;
    decoded.reserve(N);

    for (int i = 0; i < N; ++i) {
        float score = sigmoidf(scores[i]);
        if (score < minConfidence_) continue;

        int off = i * 16;
        if (off + 15 >= static_cast<int>(boxes_and_keypoints.size())) break;

        const Anchor an = (i < static_cast<int>(anchors_.size())) ? anchors_[i] : Anchor{0.5f,0.5f,1.0f,1.0f};

        float t_y = boxes_and_keypoints[off+0];
        float t_x = boxes_and_keypoints[off+1];
        float t_h = boxes_and_keypoints[off+2];
        float t_w = boxes_and_keypoints[off+3];

        float x_center = an.x_center + (t_x / x_scale) * an.w;
        float y_center = an.y_center + (t_y / y_scale) * an.h;
        float w_norm = an.w * std::exp(t_w / x_scale);
        float h_norm = an.h * std::exp(t_h / y_scale);

        float x1n = std::clamp(x_center - 0.5f*w_norm, 0.0f, 1.0f);
        float y1n = std::clamp(y_center - 0.5f*h_norm, 0.0f, 1.0f);
        float x2n = std::clamp(x_center + 0.5f*w_norm, 0.0f, 1.0f);
        float y2n = std::clamp(y_center + 0.5f*h_norm, 0.0f, 1.0f);

        int x1_t = static_cast<int>(x1n * inputWidth_);
        int y1_t = static_cast<int>(y1n * inputHeight_);
        int x2_t = static_cast<int>(x2n * inputWidth_);
        int y2_t = static_cast<int>(y2n * inputHeight_);

        int x1 = std::clamp(static_cast<int>((x1_t - pad_x) / ratio), 0, image.cols-1);
        int y1 = std::clamp(static_cast<int>((y1_t - pad_y) / ratio), 0, image.rows-1);
        int x2 = std::clamp(static_cast<int>((x2_t - pad_x) / ratio), 0, image.cols-1);
        int y2 = std::clamp(static_cast<int>((y2_t - pad_y) / ratio), 0, image.rows-1);

        int w = x2 - x1;
        int h = y2 - y1;
        if (w <=0 || h <=0) continue;

        FaceBox fb;
        fb.x = x1; fb.y = y1; fb.width = w; fb.height = h; fb.confidence = score;

        // extract 6 landmarks
        for (int k=4; k<16; k+=2) {
            float lx = boxes_and_keypoints[off+k]/x_scale*an.w + an.x_center;
            float ly = boxes_and_keypoints[off+k+1]/y_scale*an.h + an.y_center;
            int lx_img = std::clamp(static_cast<int>((lx*inputWidth_-pad_x)/ratio),0,image.cols-1);
            int ly_img = std::clamp(static_cast<int>((ly*inputHeight_-pad_y)/ratio),0,image.rows-1);
            fb.landmarks.push_back(neptune::Point{ static_cast<float>(lx_img),
                static_cast<float>(ly_img) });
}
        decoded.push_back(fb);
    }

    if (!decoded.empty()) {
        auto kept = nonMaxSuppression(decoded,0.3f,2);
        results.insert(results.end(), kept.begin(), kept.end());
    }
}

// ------------------- detectFaces -------------------
std::vector<FaceBox> FaceDetector::detectFaces(const cv::Mat& image) {
    std::vector<FaceBox> results;
    if (!engine_ || image.empty()) return results;

    cv::Mat resized = img::Preprocess::resize(image,inputWidth_,inputHeight_);
    std::vector<float> inputTensor = img::Preprocess::normalize(resized);

    if (!engine_->setInputTensor(inputTensor) || !engine_->invoke()) return results;

    int numOutputs = engine_->getNumOutputs();
    if (numOutputs==2) {
        parseMediaPipe2OutputFormat(engine_->getOutputTensor(0),
                                    engine_->getOutputTensor(1),
                                    image, results);
    } else if (numOutputs>=4) {
        parseSSDFormat(image, results);
    } else {
        parseUnknownFormat(engine_->getOutputTensor(0), image, results);
    }

    Log::info("FaceDetector","Detected "+std::to_string(results.size())+" faces");
    return results;
}
void FaceDetector::parseSSDFormat(const cv::Mat& image, std::vector<FaceBox>& results) {
    // empty for now
}

void FaceDetector::parseUnknownFormat(const std::vector<float>& output, const cv::Mat& image, std::vector<FaceBox>& results) {
    // empty for now
}


} // namespace neptune






























































   
























































































