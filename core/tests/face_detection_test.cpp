
// File: core/tests/face_detection_test.cpp
#include "neptune/FaceDetector.h"
#include "neptune/EmotionRecognizer.h"
#include "neptune/LivenessChecker.h"
#include "neptune/landmark_extractor.h"
#include "neptune/Types.h"
#include "neptune/Log.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace neptune;

// ------------------------ Helper Functions ------------------------
static std::string emotionToString(Emotion e) {
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

static std::string livenessToString(const LivenessResult& live) {
    if (live.status == LivenessStatus::LIVE)
        return "LIVE (" + live.reason + ", conf=" + std::to_string(live.confidence) + ")";
    if (live.status == LivenessStatus::NOT_LIVE)
        return "NOT LIVE (" + live.reason + ", conf=" + std::to_string(live.confidence) + ")";
    return "UNKNOWN";
}

static cv::Scalar livenessColor(const LivenessResult& live) {
    if (live.status == LivenessStatus::LIVE) return {0, 255, 0};     // green
    if (live.status == LivenessStatus::NOT_LIVE) return {0, 0, 255}; // red
    return {0, 255, 255}; // yellow
}

static cv::Scalar emotionColor(Emotion emotion) {
    switch (emotion) {
        case Emotion::HAPPINESS: return {0, 255, 0};    // Green for happy
        case Emotion::NEUTRAL: return {255, 255, 0};    // Yellow for neutral
        case Emotion::SURPRISE: return {255, 165, 0};   // Orange for surprise
        default: return {0, 0, 255};                   // Red for negative emotions
    }
}

static cv::Rect clampRect(const cv::Rect& r, const cv::Size& sz) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::max(1, std::min(r.width, sz.width - x));
    int h = std::max(1, std::min(r.height, sz.height - y));
    return {x, y, w, h};
}

static void drawLandmarks(cv::Mat& image, const std::vector<Point>& landmarks, const cv::Scalar& color) {
    for (const auto& point : landmarks) {
        cv::circle(image, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)), 2, color, -1);
    }
}

// ------------------------ Main Function ------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " --image <path> [--backend <0=tflite|1=mediapipe|2=auto>] [--fps] [--debug]\n"
                  << "  " << argv[0] << " --video [--backend <0|1|2>] [--fps] [--debug]\n";
        return 1;
    }

    // Default model paths (adjust paths if your build system places models elsewhere)
    const std::string faceModelPath = "../../models/face_detection_short_range.tflite";
    const std::string landmarkModelPath = "../../models/face_landmark.tflite";
    const std::string emotionModelPath = "../../models/mobilenet_emotion.tflite";

    NeptuneConfig config;
    config.faceDetectionModelPath = faceModelPath;
    config.emotionModelPath = emotionModelPath;
    config.faceDetectorBackend = FaceDetectorBackend::MEDIAPIPE;
    config.earClosedThreshold = 0.25f;
    config.blinkMinFrames = 2;
    config.headYawChangeMinDeg = 20.0f;
    config.headPitchChangeMinDeg = 15.0f;
    config.livenessWindowMs = 3000.0;

    bool showFps = false;
    bool debugMode = false;
    bool videoMode = false;
    std::string imagePath;

    // ------------------------ Parse Command Line ------------------------
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            int backend = std::atoi(argv[++i]);
            if (backend >= 0 && backend <= 2) {
                config.faceDetectorBackend = static_cast<FaceDetectorBackend>(backend);
            }
        } else if (arg == "--fps") {
            showFps = true;
        } else if (arg == "--debug") {
            debugMode = true;
        } else if (arg == "--image" && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (arg == "--video") {
            videoMode = true;
        }
    }

    if (debugMode) {
        std::cout << "Debug logging enabled\n";
    }

    std::cout << "Initializing Neptune Facial SDK...\n";
    switch (config.faceDetectorBackend) {
        case FaceDetectorBackend::TFLITE: std::cout << "Backend: TFLite\n"; break;
        case FaceDetectorBackend::MEDIAPIPE: std::cout << "Backend: MediaPipe\n"; break;
        case FaceDetectorBackend::AUTO: std::cout << "Backend: Auto\n"; break;
    }

    auto detector = FaceDetector::create(faceModelPath, config);
    auto emo = EmotionRecognizer::create(emotionModelPath, config);
    LivenessChecker liveness(config);
    LandmarkExtractor landmarkExtractor(landmarkModelPath);

    if (!detector || !emo) {
        std::cerr << "ERROR: Failed to initialize detector or emotion recognizer.\n";
        return 1;
    }

    std::cout << "Neptune SDK initialized successfully!\n";

    // Set video mode based on input
    liveness.setVideoMode(videoMode);

    // Decide mode
    if (!imagePath.empty() && !videoMode) {
        // IMAGE MODE
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: failed to load " << imagePath << "\n";
            return 1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto faces = detector->detectFaces(image);
        auto end = std::chrono::high_resolution_clock::now();
        double detectionTime = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Detected " << faces.size() << " faces in " << detectionTime << " ms\n";

        cv::Mat displayImage = image.clone();
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& f = faces[i];
            cv::Rect r = clampRect(cv::Rect(f.x, f.y, f.width, f.height), image.size());

            // Use Process(frame, rect) when the LandmarkExtractor expects a full-image + rect
            auto landmarks2D = landmarkExtractor.Process(image, r);

            // The extractor returns absolute image coordinates (since we passed the full image + rect)
            faces[i].landmarks.clear();
            for (auto& p : landmarks2D) {
                faces[i].landmarks.push_back(Point{p.x, p.y});
            }

            std::cout << "Face " << i+1
                      << " | Box: (" << r.x << "," << r.y << "," << r.width << "," << r.height << ")"
                      << " | Landmarks: " << faces[i].landmarks.size() << "\n";

            cv::rectangle(displayImage, r, cv::Scalar(0, 255, 0), 2);
            drawLandmarks(displayImage, faces[i].landmarks, cv::Scalar(255, 255, 0));

            if (!faces[i].landmarks.empty()) {
                // For emotion/liveness we pass the cropped face ROI
                cv::Mat faceROI = image(r).clone();
                auto er = emo->predictEmotion(faceROI);
                auto live = liveness.check(faces[i]); // Will immediately return NOT_LIVE for static images

                std::string infoText = emotionToString(er.emotion) + " | " + livenessToString(live);
                cv::putText(displayImage, infoText, cv::Point(r.x, std::max(0, r.y - 15)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, livenessColor(live), 2);
            }
        }

        cv::imshow("Neptune Facial SDK - Image Test", displayImage);
        cv::waitKey(0);
    } else if (videoMode) {
        
        // VIDEO MODE
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: failed to open camera or video file\n";
            return 1;
        }

        std::cout << "Video capture started. Press ESC to exit.\n";

        cv::Mat frame;
        int frameCounter = 0;
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            frameCounter++;

            auto start = std::chrono::high_resolution_clock::now();
            auto faces = detector->detectFaces(frame);
            auto end = std::chrono::high_resolution_clock::now();
            double detectionTime = std::chrono::duration<double, std::milli>(end - start).count();

            cv::Mat displayImage = frame.clone();
            for (size_t i = 0; i < faces.size(); ++i) {
                const auto& f = faces[i];
                cv::Rect r = clampRect(cv::Rect(f.x, f.y, f.width, f.height), frame.size());

                // Match the LandmarkExtractor signature: pass full frame + rect so returned points are absolute
                auto landmarks2D = landmarkExtractor.Process(frame, r);

                faces[i].landmarks.clear();
                for (auto& p : landmarks2D) {
                    faces[i].landmarks.push_back(Point{p.x, p.y});
                }

                std::cout << "Frame " << frameCounter << " - Face " << i+1
                          << " | Box: (" << r.x << "," << r.y << "," << r.width << "," << r.height << ")"
                          << " | Landmarks: " << faces[i].landmarks.size() << "\n";

                cv::rectangle(displayImage, r, cv::Scalar(0, 255, 0), 2);
                drawLandmarks(displayImage, faces[i].landmarks, cv::Scalar(255, 255, 0));

                if (!faces[i].landmarks.empty()) {
                    cv::Mat faceROI = frame(r).clone();
                    auto er = emo->predictEmotion(faceROI);
                    auto live = liveness.check(faces[i]); // Will use proper temporal tracking for video

                    std::string infoText = emotionToString(er.emotion) + " | " + livenessToString(live);
                    cv::putText(displayImage, infoText, cv::Point(r.x, std::max(0, r.y - 15)),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, livenessColor(live), 2);
                }
            }

            if (showFps && detectionTime > 0.0) {
                double fps = 1000.0 / detectionTime;
                std::string fpsText = "Detection: " + std::to_string(fps) + " FPS";
                cv::putText(displayImage, fpsText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);
            }

            cv::imshow("Neptune Facial SDK - Video Test", displayImage);

            int key = cv::waitKey(1);
            if (key == 27) break; 
        }

        cap.release();
        cv::destroyAllWindows();
    } else {
        std::cerr << "Error: no mode selected. Use --image <path> or --video\n";
        return 1;
    }

    std::cout << "Test completed successfully!\n";
    return 0;
}





// Head movement detection is too sensitive - It's detecting noise as head movements
// Blink detection isn't working - It's not detecting your actual blinks
// Liveness is being confirmed without actual proof - It should require both blinks AND head movements



// EAR is huge (~1.5–1.8) instead of ~0.2..0.3 → your computeEAR() is being fed the eye landmarks in the wrong order (so the horizontal / vertical pairs are incorrect), which makes the ratio large and breaks blink detection.
// Head-movement counter keeps rising even when the user is still → pose jitter/noise is being treated as instantaneous movement every frame (no warm-up baseline, no smoothing, no minimum interval).
// Blink count stays zero and the system reports LIVE prematurely → because EAR was wrong + liveness logic marks proven too easily when the above counters are noisy.



// Face detection: MediaPipe (via TFLite).
// Emotion recognition: TFLite MobileNet model.
// Liveness detection: Custom implementation based on facial landmarks (blinks, head turns).


// ./tests/face_detection_test --video
// ./tests/face_detection_test --image ../tests/assets/angry.jpg


//to see how the sdk is organised, run: 
// tree -I 'build|bin|.vscode' --dirsfirst



//create and activate a virtual environment:

//python3 -m venv venv
//source venv/bin/activate

//Reinstall pybind11 inside this venv

//pip install --upgrade pip
//pip install pybind11



//bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/examples/desktop/face_mesh:face_mesh_cpu --verbose_failures

  //Bazel will place the executable here:

  //bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu
//   is a MediaPipe executable for running the Face Mesh graph on CPU. Its main purpose is:
//   Detect faces in images or video.
//   Extract 468 3D facial landmarks (Face Mesh) per detected face.
//   Optionally, you can feed it into downstream calculators like emotion recognition, liveness detection, or gesture detection, depending on the graph you load (pbtxt).
//   So, this executable is a runtime for your MediaPipe graph. You pass a .pbtxt calculator graph file and input images/video.





//   -sh * | sort -h

//du -sh third_party/* | sort -h
