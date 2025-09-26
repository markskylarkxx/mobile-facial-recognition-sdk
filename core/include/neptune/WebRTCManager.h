// WebRTCManager.h
#pragma once

#include <opencv2/opencv.hpp>
#include "FaceDetector.h"
#include "EmotionRecognizer.h"
#include "LivenessChecker.h"
#include "landmark_extractor.h"
#include "Types.h"

class WebRTCManager {
public:
    WebRTCManager();
    ~WebRTCManager();

    // Start the WebRTC server and wait for a connection
    bool start();

    // The most important function: WebRTC will call this when a new frame arrives
    void onFrameReceived(cv::Mat &frame);
    void testWithLocalWebcam(); // Add this declaration

private:
    // Your AI objects from main.cpp
    std::unique_ptr<neptune::FaceDetector> detector;
    std::unique_ptr<neptune::EmotionRecognizer> emo;
    neptune::LivenessChecker liveness;
    LandmarkExtractor landmarkExtractor;
    neptune::NeptuneConfig config;

    // Helper functions from your main.cpp
    std::string emotionToString(neptune::Emotion e);
    std::string livenessToString(const neptune::LivenessResult& live);
    cv::Rect clampRect(const cv::Rect& r, const cv::Size& sz);
    void drawLandmarks(cv::Mat& image, const std::vector<neptune::Point>& landmarks, const cv::Scalar& color);
};