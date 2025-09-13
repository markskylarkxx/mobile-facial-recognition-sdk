//
// File: NeptuneFacialSDK/core/tests/real_time_detection_test.cpp
//
// A real-time demo application for FaceDetector using a webcam.
//

#include "neptune/FaceDetector.h"
#include "neptune/Types.h"
#include "neptune/Log.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace neptune;

int main(int argc, char** argv) {
    std::cout << "==== Neptune Real-Time Face Detection Test ====" << std::endl;

    // --- Step 1: Locate and load the model ---
    const std::string modelPath = "../../models/face_detection_short_range.tflite";

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "ERROR: Could not find model at: " << modelPath << std::endl;
        return 1;
    }

    // --- Step 2: Load config ---
    NeptuneConfig config;
    config.faceDetectionModelPath = modelPath;
    config.minFaceDetectionConfidence = 0.5f; // Set a reasonable confidence threshold

    // --- Step 3: Create FaceDetector ---
    auto detector = FaceDetector::create(modelPath, config);
    if (!detector) {
        std::cerr << "ERROR: Failed to create FaceDetector." << std::endl;
        return 1;
    }

    // --- Step 4: Open the default camera ---
    cv::VideoCapture cap(0); // 0 is the default camera index
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Failed to open webcam." << std::endl;
        return 1;
    }
    
    // --- Step 5: Start the video processing loop ---
    cv::Mat frame;
    std::cout << "Press 'q' to exit the video stream." << std::endl;

    while (true) {
        // Read a new frame from the camera
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Captured empty frame." << std::endl;
            break;
        }

        // Run face detection on the current frame
        std::vector<FaceBox> faces = detector->detectFaces(frame);

        // Draw bounding boxes for visual feedback
        for (const auto& f : faces) {
            cv::rectangle(frame, cv::Rect(f.x, f.y, f.width, f.height), cv::Scalar(0, 255, 0), 2);
        }

        // Display the frame with detections in a window
        cv::imshow("Neptune Face Detection", frame);

        // Break the loop if the 'q' key is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // --- Step 6: Clean up ---
    cap.release();
    cv::destroyAllWindows();

    return 0;
}