#include "neptune/WebRTCManager.h"
#include <iostream>
#include <chrono>

using namespace neptune;

WebRTCManager::WebRTCManager()
    : config(), // Initialize config (if it has a default constructor)
      detector(nullptr), // Initialize unique_ptr to nullptr
      emo(nullptr), // Initialize unique_ptr to nullptr
      liveness(config), // Initialize liveness with config
      landmarkExtractor("../../models/face_landmark.tflite") // Initialize landmarkExtractor
{
    // 1. Move your configuration code here
    const std::string faceModelPath = "../../models/face_detection_short_range.tflite";
    const std::string landmarkModelPath = "../../models/face_landmark.tflite";
    const std::string emotionModelPath = "../../models/mobilenet_emotion.tflite";

    config.faceDetectionModelPath = faceModelPath;
    config.emotionModelPath = emotionModelPath;
    config.faceDetectorBackend = FaceDetectorBackend::MEDIAPIPE;
    config.earClosedThreshold = 0.25f;
    config.blinkMinFrames = 2;
    config.headYawChangeMinDeg = 20.0f;
    config.headPitchChangeMinDeg = 15.0f;
    config.livenessWindowMs = 3000.0;

    // 2. Move your AI initialization code here
    detector = FaceDetector::create(faceModelPath, config);
    emo = EmotionRecognizer::create(emotionModelPath, config);
    LandmarkExtractor landmarkExtractor(landmarkModelPath);

    // 3. Initialize Liveness checker for video mode
    liveness.setVideoMode(true); // We are in video mode

    if (!detector || !emo) {
        std::cerr << "ERROR: Failed to initialize detector or emotion recognizer.\n";
        // Handle error
    }
}

WebRTCManager::~WebRTCManager() {
    // Cleanup code will go here later
}

bool WebRTCManager::start() {
    std::cout << "WebRTC server starting... (Not implemented yet)" << std::endl;
    return true;
}

void WebRTCManager::onFrameReceived(cv::Mat &frame) {
    // This is the exact same logic from your video loop, but now it processes a frame from the phone
    auto faces = detector->detectFaces(frame);

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& f = faces[i];
        cv::Rect r = clampRect(cv::Rect(f.x, f.y, f.width, f.height), frame.size());

        auto landmarks2D = landmarkExtractor.Process(frame, r);
        faces[i].landmarks.clear();
        for (auto& p : landmarks2D) {
            faces[i].landmarks.push_back(Point{p.x, p.y});
        }

        if (!faces[i].landmarks.empty()) {
            cv::Mat faceROI = frame(r).clone();
            auto er = emo->predictEmotion(faceROI);
            auto live = liveness.check(faces[i]);

            std::cout << "RESULT FOR PHONE: Emotion=" << emotionToString(er.emotion)
                      << " | Liveness=" << livenessToString(live) << std::endl;
        }
    }
}

std::string WebRTCManager::emotionToString(neptune::Emotion e) {
    switch (e) {
        case Emotion::ANGER: return "ANGER";
        case Emotion::DISGUST: return "DISGUST";
        case Emotion::FEAR: return "FEAR";
        case Emotion::HAPPINESS: return "HAPPY";
        case Emotion::SADNESS: return "SAD";
        case Emotion::SURPRISE: return "SURPRISE";
        case Emotion::NEUTRAL: return "NEUTRAL";
        default: return "UNKNOWN";
    }
}

std::string WebRTCManager::livenessToString(const neptune::LivenessResult& live) {
    if (live.status == LivenessStatus::LIVE)
        return "LIVE (" + live.reason + ", conf=" + std::to_string(live.confidence) + ")";
    if (live.status == LivenessStatus::NOT_LIVE)
        return "NOT LIVE (" + live.reason + ", conf=" + std::to_string(live.confidence) + ")";
    return "UNKNOWN";
}

cv::Rect WebRTCManager::clampRect(const cv::Rect& r, const cv::Size& sz) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::max(1, std::min(r.width, sz.width - x));
    int h = std::max(1, std::min(r.height, sz.height - y));
    return {x, y, w, h};
}

void WebRTCManager::drawLandmarks(cv::Mat& image, const std::vector<neptune::Point>& landmarks, const cv::Scalar& color) {
    for (const auto& point : landmarks) {
        cv::circle(image, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)), 2, color, -1);
    }
}

// WebRTCManager.cpp
void WebRTCManager::testWithLocalWebcam() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: failed to open camera" << std::endl;
        return;
    }

    cv::Mat testFrame;
    std::cout << "Testing with local webcam. Press 'q' to quit." << std::endl;

    while (true) {
        cap >> testFrame;
        if (testFrame.empty()) break;

        // Simulate WebRTC calling our function!
        this->onFrameReceived(testFrame);

        // Optional: Display the frame to see something
        cv::imshow("Test Preview", testFrame);
        if (cv::waitKey(1) == 'q') break;
    }
    cap.release();
    cv::destroyAllWindows();
}