
// File: core/include/neptune/LivenessChecker.h
#pragma once

#include "neptune/Types.h"
#include <deque>
#include <chrono>
#include <string>

namespace neptune {

class LivenessChecker {
public:
    explicit LivenessChecker(const NeptuneConfig& config);

    // Main entry point: takes a face with landmarks, returns liveness result
    LivenessResult check(const FaceBox& face);

    //added!!
    void setVideoMode(bool enabled);
    //added!
    void reset(); // Reset all tracking state
    
    
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

    // Helpers
    float computeEAR(const std::vector<Point>& eyeLandmarks);
    float estimateHeadYaw(const std::vector<Point>& landmarks);
    float estimateHeadPitch(const std::vector<Point>& landmarks);
    
    // New helper methods for better detection
    bool detectBlink(float currentEAR);
    bool detectHeadMovement(float currentYaw, float currentPitch);

    bool isInitialized_;
    int frameCount_;
    bool isVideoMode_;

};

} // namespace neptune





