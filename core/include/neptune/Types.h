
// // File: NeptuneFacialSDK/core/include/neptune/Types.h
// //
// // Core data structures and enumerations for the Neptune Facial SDK's C++ engine.
// // This ensures consistency between components like face detection, emotion recognition, and configuration.

// #pragma once

// #include <vector>
// #include <string>

// namespace neptune {

// // Represents a bounding box anchor (for detection algorithms like SSD).
// struct Anchor {
//     float x_center;
//     float y_center;
//     float w;
//     float h;
// };

// // Represents a 2D point, typically for facial landmarks.
// struct Point {
//     float x;
//     float y;
// };

// // Represents a bounding box around a detected face.
// struct FaceBox {
//     int x;
//     int y;
//     int width;
//     int height;
//     float confidence;
//     std::vector<Point> landmarks;      // 68 or 106 facial landmarks


// };

// // Corrected emotion enum matching model output
// enum class Emotion {
//     ANGER = 0,
//     DISGUST = 1,
//     FEAR = 2,
//     HAPPINESS = 3,
//     SADNESS = 4,
//     SURPRISE = 5,
//     NEUTRAL = 6,
//     UNKNOWN = 7
// };




// // Represents the result of an emotion recognition prediction.
// struct EmotionResult {
//     Emotion emotion;
//     float confidence;
// };

// // Defines the possible outcomes of the liveness check.
// enum class LivenessStatus {
//     UNKNOWN,
//     NOT_LIVE,
//     LIVE
// };

// struct LivenessResult {
//     LivenessStatus status;
//     float confidence; 
//     std::string reason;   // NEW: explains WHY (blink / head turn / none)

// };

// // Represents a single frame analysis result, combining all predictions.
// struct NeptuneResult {
//     bool hasFace;
//     FaceBox faceBox;
//     EmotionResult emotion;
//     LivenessResult liveness;
//     // std::vector<Point> landmarks;
// };

// // Represents the result of processing a single detected face.
// struct ProcessedFace {
//     FaceBox faceBox;
//     EmotionResult emotion;
//     LivenessResult liveness;
// };

// // Configuration settings for the SDK.
// struct NeptuneConfig {
//     std::string faceDetectionModelPath;
//     std::string emotionModelPath;
//     std::string livenessModelPath;
//     std::string faceLandmarkModelPath; // for 468 landmark mode

//     float minFaceDetectionConfidence = 0.5f;
//     float minEmotionConfidence = 0.20f;

//     // Liveness-specific settings
//     float earClosedThreshold = 0.20f;       // EAR threshold for eye closed
//     int blinkMinFrames = 2;                 // frames needed for a blink
//     float headYawChangeMinDeg = 10.0f;      // yaw threshold
//     float headPitchChangeMinDeg = 8.0f;     // pitch threshold
//     double livenessWindowMs = 2000.0;       // time window for pose changes

// };

// } // namespace neptune













//...........................................................................


// File: NeptuneFacialSDK/core/include/neptune/Types.h
#pragma once

#include <vector>
#include <string>
#include <chrono>

namespace neptune {

// Represents a bounding box anchor (for detection algorithms like SSD).
struct Anchor {
    float x_center;
    float y_center;
    float w;
    float h;
};

// Represents a 2D point, typically for facial landmarks.
struct Point {
    float x;
    float y;
    
    Point() : x(0), y(0) {}
    Point(float x, float y) : x(x), y(y) {}
};

// Represents a bounding box around a detected face.
struct FaceBox {
    int x;
    int y;
    int width;
    int height;
    float confidence;
    std::vector<Point> landmarks; // 68, 106, or 468 facial landmarks
    std::chrono::steady_clock::time_point detectionTime;
    
    FaceBox() : x(0), y(0), width(0), height(0), confidence(0.0f) {}
};

// Corrected emotion enum matching model output
enum class Emotion {
    ANGER = 0,
    DISGUST = 1,
    FEAR = 2,
    HAPPINESS = 3,
    SADNESS = 4,
    SURPRISE = 5,
    NEUTRAL = 6,
    UNKNOWN = 7
};

// Represents the result of an emotion recognition prediction.
struct EmotionResult {
    Emotion emotion;
    float confidence;
    std::vector<float> probabilities; // All emotion probabilities
    
    EmotionResult() : emotion(Emotion::UNKNOWN), confidence(0.0f) {}
    EmotionResult(Emotion e, float conf) : emotion(e), confidence(conf) {}

};

// Defines the possible outcomes of the liveness check.
enum class LivenessStatus {
    UNKNOWN,
    NOT_LIVE,
    LIVE
};

struct LivenessResult {
    LivenessStatus status;
    float confidence; 
    std::string reason;
    
    LivenessResult() : status(LivenessStatus::UNKNOWN), confidence(0.0f) {}
};

// Represents a single frame analysis result, combining all predictions.
struct NeptuneResult {
    bool hasFace;
    FaceBox faceBox;
    EmotionResult emotion;
    LivenessResult liveness;
    double processingTimeMs;
    
    NeptuneResult() : hasFace(false), processingTimeMs(0.0) {}
};

// Represents the result of processing a single detected face.
struct ProcessedFace {
    FaceBox faceBox;
    EmotionResult emotion;
    LivenessResult liveness;
    
    ProcessedFace() {}
};

// Face detector backend options
enum class FaceDetectorBackend {
    TFLITE = 0,
    MEDIAPIPE = 1,
    AUTO = 2
};


// Configuration settings for the SDK.
struct NeptuneConfig {
    std::string faceDetectionModelPath;
    std::string emotionModelPath;
    std::string livenessModelPath;
    std::string faceLandmarkModelPath;

    float minFaceDetectionConfidence = 0.5f;
    float minEmotionConfidence = 0.20f;

    // Liveness-specific settings
    float earClosedThreshold = 0.20f;
    int blinkMinFrames = 2;
    float headYawChangeMinDeg = 10.0f;
    float headPitchChangeMinDeg = 8.0f;
    double livenessWindowMs = 2000.0;

    // MediaPipe configuration
    FaceDetectorBackend faceDetectorBackend = FaceDetectorBackend::AUTO;
    bool useMediaPipe = true;
    int maxFaces = 2;
    int landmarkType = 468; // 68, 106, or 468 landmarks
    
    // Performance settings
    int processingWidth = 320;
    int processingHeight = 240;
    bool enableGPU = false;
};

struct NormalizedRect {
    float x_center = 0.0f;
    float y_center = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float rotation = 0.0f; // radians
};


} // namespace neptune