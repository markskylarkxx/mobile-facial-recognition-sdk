
#pragma once

#include "neptune/Types.h"
#include <deque>
#include <chrono>
#include <string>
#include <vector>

namespace neptune {

class LivenessChecker {
public:
    explicit LivenessChecker(const NeptuneConfig& config);

    // Main entry point: takes a face with landmarks, returns liveness result
    LivenessResult check(const FaceBox& face);

    // Video mode control
    void setVideoMode(bool enabled);

    // Resetters
    void resetForNewFrame();

private:
    NeptuneConfig config_;

    // State tracking for blink detection
    std::deque<float> earHistory_;
    int blinkFrameCount_ = 0;

    // State tracking for head movement detection
    float lastYaw_ = 0.0f;
    float lastPitch_ = 0.0f;
    std::chrono::steady_clock::time_point lastHeadMoveTime_;

    // Smoothed pose values (these were missing)
    float smoothedYaw_ = 0.0f;
    float smoothedPitch_ = 0.0f;

    // Helpers
    float computeEAR(const std::vector<Point>& eyeLandmarks);
    float estimateHeadYaw(const std::vector<Point>& landmarks);
    float estimateHeadPitch(const std::vector<Point>& landmarks);

    // New helper methods for detection
    bool detectBlink(float currentEAR);
    bool detectHeadMovement(float currentYaw, float currentPitch);

    bool isInitialized_;
    int frameCount_;
    bool isVideoMode_;

    // NEW ANTI-SPOOFING MEMBERS
    bool hasProvenLiveness_;                                 // Has the face proven it's alive?
    std::chrono::steady_clock::time_point firstDetectionTime_; // When we first detected this face
    int totalBlinksDetected_ = 0;                            // Total blinks detected for this face
    int totalHeadMovements_ = 0;                             // Total head movements detected for this face

    // NOISE FILTERING MEMBERS
    std::deque<float> yawHistory_;              // History of yaw values for trend analysis
    std::deque<float> pitchHistory_;            // History of pitch values for trend analysis
    int stableFrameCount_ = 0;                  // Frames without significant movement

    // Warm-up and debounce parameters (tunable)
    int poseWarmupFrames_ = 3;                  // frames used to stabilize initial pose
    int poseWarmupCount_ = 0;                   // current warm-up counter
    int minMovementIntervalMs_ = 300;           // minimum ms between counted head movements

    float baselineEAR_; // Added for calibration
    int calibrationFrames_; // Added for calibration
};

} // namespace neptune
