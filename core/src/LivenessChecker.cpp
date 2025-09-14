
#include "neptune/LivenessChecker.h"
#include "neptune/MediaPipeLandmarks.h"
#include "neptune/Log.h"
#include <cmath>

using namespace neptune;

LivenessChecker::LivenessChecker(const NeptuneConfig& config)
    : config_(config),
      lastHeadMoveTime_(std::chrono::steady_clock::now()),
      blinkFrameCount_(0),
      lastYaw_(0.0f),
      lastPitch_(0.0f),
      isInitialized_(false),
      frameCount_(0),
      isVideoMode_(false) {}  // Initialize isVideoMode_

void LivenessChecker::setVideoMode(bool enabled) {
    isVideoMode_ = enabled;
    Log::info("LivenessChecker", "Video mode set to: " + std::string(enabled ? "true" : "false"));
    
    // Reset state when mode changes
    if (!enabled) {
        resetForNewFrame();
    }
}

void LivenessChecker::resetForNewFrame() {
    // Reset frame-specific state
    frameCount_ = 0;
    isInitialized_ = false;
    earHistory_.clear();
    blinkFrameCount_ = 0;
    lastHeadMoveTime_ = std::chrono::steady_clock::now();
    lastYaw_ = 0.0f;
    lastPitch_ = 0.0f;
    
    Log::debug("LivenessChecker", "Reset for new frame/image");
}

float LivenessChecker::computeEAR(const std::vector<Point>& eyeLandmarks) {
    // MediaPipe eye landmarks: 6 points per eye in this order:
    // [0] = outer corner, [1] = top-outer, [2] = top-inner, [3] = inner corner, [4] = bottom-inner, [5] = bottom-outer
    
    if (eyeLandmarks.size() != 6) {
        Log::warn("LivenessChecker", "Invalid eye landmarks count: " + std::to_string(eyeLandmarks.size()));
        return -1.0f;
    }

    try {
        // Calculate EAR using the 6-point formula
        // EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        float vertical1 = std::sqrt(std::pow(eyeLandmarks[1].x - eyeLandmarks[5].x, 2) + 
                                   std::pow(eyeLandmarks[1].y - eyeLandmarks[5].y, 2));
        float vertical2 = std::sqrt(std::pow(eyeLandmarks[2].x - eyeLandmarks[4].x, 2) + 
                                   std::pow(eyeLandmarks[2].y - eyeLandmarks[4].y, 2));
        float horizontal = std::sqrt(std::pow(eyeLandmarks[0].x - eyeLandmarks[3].x, 2) + 
                                    std::pow(eyeLandmarks[0].y - eyeLandmarks[3].y, 2));
        
        if (horizontal < 1e-6f) {
            Log::warn("LivenessChecker", "Horizontal eye distance too small for EAR calculation");
            return -1.0f;
        }
        
        float ear = (vertical1 + vertical2) / (2.0f * horizontal);
        return std::max(0.0f, std::min(1.0f, ear));
        
    } catch (const std::exception& e) {
        Log::error("LivenessChecker", "EAR calculation error: " + std::string(e.what()));
        return -1.0f;
    }
}

float LivenessChecker::estimateHeadYaw(const std::vector<Point>& landmarks) {
    if (landmarks.size() != 468) {
        Log::warn("LivenessChecker", "Invalid landmarks count for MediaPipe: " + std::to_string(landmarks.size()));
        return 0.0f;
    }

    try {
        // Get nose tip
        const Point& noseTip = landmarks[1];
        
        // Left eye center
        Point leftEyeCenter = {0.0f, 0.0f};
        std::vector<int> leftEyeIndices = {362, 382, 381, 380, 374, 373};
        for (int idx : leftEyeIndices) {
            leftEyeCenter.x += landmarks[idx].x;
            leftEyeCenter.y += landmarks[idx].y;
        }
        leftEyeCenter.x /= leftEyeIndices.size();
        leftEyeCenter.y /= leftEyeIndices.size();
        
        // Right eye center
        Point rightEyeCenter = {0.0f, 0.0f};
        std::vector<int> rightEyeIndices = {33, 7, 163, 144, 145, 153};
        for (int idx : rightEyeIndices) {
            rightEyeCenter.x += landmarks[idx].x;
            rightEyeCenter.y += landmarks[idx].y;
        }
        rightEyeCenter.x /= rightEyeIndices.size();
        rightEyeCenter.y /= rightEyeIndices.size();
        
        Log::debug("LivenessChecker", "Left eye center: (" + std::to_string(leftEyeCenter.x) + ", " + std::to_string(leftEyeCenter.y) + ")");
        Log::debug("LivenessChecker", "Right eye center: (" + std::to_string(rightEyeCenter.x) + ", " + std::to_string(rightEyeCenter.y) + ")");
        
        // Calculate face center between eyes
        float eyesCenterX = (leftEyeCenter.x + rightEyeCenter.x) / 2.0f;
        float eyesDistance = std::abs(rightEyeCenter.x - leftEyeCenter.x);
        
        if (eyesDistance < 1e-3f) {
            Log::warn("LivenessChecker", "Eyes too close for yaw calculation: " + std::to_string(eyesDistance));
            return 0.0f;
        }
        
        float delta = noseTip.x - eyesCenterX;
        float normalizedYaw = delta / eyesDistance;
        
        return std::max(-1.0f, std::min(1.0f, normalizedYaw));
        
    } catch (const std::exception& e) {
        Log::error("LivenessChecker", "Head yaw estimation error: " + std::string(e.what()));
        return 0.0f;
    }
}

float LivenessChecker::estimateHeadPitch(const std::vector<Point>& landmarks) {
    if (landmarks.size() != 468) {
        Log::warn("LivenessChecker", "Invalid landmarks count for MediaPipe: " + std::to_string(landmarks.size()));
        return 0.0f;
    }

    try {
        const int NOSE_TIP = 1;
        const int FOREHEAD = 10;
        const int CHIN = 175;
        
        const Point& noseTip = landmarks[NOSE_TIP];
        const Point& forehead = landmarks[FOREHEAD];
        const Point& chin = landmarks[CHIN];
        
        Log::debug("LivenessChecker", "Pitch landmarks:");
        Log::debug("LivenessChecker", "  Nose [1]: (" + std::to_string(noseTip.x) + ", " + std::to_string(noseTip.y) + ")");
        Log::debug("LivenessChecker", "  Forehead [10]: (" + std::to_string(forehead.x) + ", " + std::to_string(forehead.y) + ")");
        Log::debug("LivenessChecker", "  Chin [175]: (" + std::to_string(chin.x) + ", " + std::to_string(chin.y) + ")");
        
        float faceHeight = std::abs(chin.y - forehead.y);
        if (faceHeight < 1e-3f) {
            Log::warn("LivenessChecker", "Face height too small for pitch calculation: " + std::to_string(faceHeight));
            return 0.0f;
        }
        
        float faceCenterY = (forehead.y + chin.y) / 2.0f;
        float normalizedPitch = (noseTip.y - faceCenterY) / faceHeight;
        
        return std::max(-1.0f, std::min(1.0f, normalizedPitch));
        
    } catch (const std::exception& e) {
        Log::error("LivenessChecker", "Head pitch estimation error: " + std::string(e.what()));
        return 0.0f;
    }
}

bool LivenessChecker::detectBlink(float currentEAR) {
    if (currentEAR < 0.0f) {
        Log::warn("LivenessChecker", "Invalid EAR value, skipping blink detection");
        return false;
    }
    
    earHistory_.push_back(currentEAR);
    if (earHistory_.size() > 10) {
        earHistory_.pop_front();
    }
    
    Log::debug("LivenessChecker", "Current EAR: " + std::to_string(currentEAR) + 
               ", Threshold: " + std::to_string(config_.earClosedThreshold) +
               ", Blink frames: " + std::to_string(blinkFrameCount_));
    
    if (currentEAR < config_.earClosedThreshold) {
        blinkFrameCount_++;
        Log::debug("LivenessChecker", "Eyes closed, frame count: " + std::to_string(blinkFrameCount_));
        return false;
    } else {
        if (blinkFrameCount_ >= config_.blinkMinFrames) {
            Log::info("LivenessChecker", "Blink detected! Closed for " + std::to_string(blinkFrameCount_) + " frames");
            blinkFrameCount_ = 0;
            return true;
        }
        blinkFrameCount_ = 0;
        return false;
    }
}

bool LivenessChecker::detectHeadMovement(float currentYaw, float currentPitch) {
    if (!isInitialized_) {
        lastYaw_ = currentYaw;
        lastPitch_ = currentPitch;
        isInitialized_ = true;
        Log::info("LivenessChecker", "Initialized head pose tracking - Yaw: " + std::to_string(currentYaw) + 
                 ", Pitch: " + std::to_string(currentPitch));
        return false;
    }
    
    float yawChange = std::abs(currentYaw - lastYaw_);
    float pitchChange = std::abs(currentPitch - lastPitch_);
    
    float yawDegrees = yawChange * 45.0f;
    float pitchDegrees = pitchChange * 45.0f;
    
    Log::debug("LivenessChecker", "Head movement: Yaw change=" + std::to_string(yawDegrees) + 
               "°, Pitch change=" + std::to_string(pitchDegrees) + "°");
    
    bool movementDetected = (yawDegrees > config_.headYawChangeMinDeg) || 
                           (pitchDegrees > config_.headPitchChangeMinDeg);
    
    if (movementDetected) {
        lastHeadMoveTime_ = std::chrono::steady_clock::now();
        Log::info("LivenessChecker", "Head movement detected: Yaw=" + std::to_string(yawDegrees) + 
                 "°, Pitch=" + std::to_string(pitchDegrees) + "°");
    }
    
    return movementDetected;
}

LivenessResult LivenessChecker::check(const FaceBox& face) {
    LivenessResult result;
    frameCount_++;
    
    // EARLY EXIT: For static images, always return NOT_LIVE without processing
    if (!isVideoMode_) {
        result.status = LivenessStatus::NOT_LIVE;
        result.confidence = 0.95f;
        result.reason = "Static image - no temporal data available";
        Log::info("LivenessChecker", "Static image detected - marked as NOT_LIVE");
        return result;
    }
    
    if (face.landmarks.empty()) {
        result.status = LivenessStatus::UNKNOWN;
        result.confidence = 0.0f;
        result.reason = "No landmarks available";
        return result;
    }

    if (face.landmarks.size() != 468) {
        result.status = LivenessStatus::UNKNOWN;
        result.confidence = 0.0f;
        result.reason = "Invalid landmark count: " + std::to_string(face.landmarks.size()) + " (expected 468)";
        Log::error("LivenessChecker", result.reason);
        return result;
    }

    try {
        // Extract eyes
        std::vector<Point> leftEye;
        std::vector<int> leftEyeIndices = {362, 382, 381, 380, 374, 373};
        for (int idx : leftEyeIndices) {
            leftEye.push_back(face.landmarks[idx]);
        }
        
        std::vector<Point> rightEye;
        std::vector<int> rightEyeIndices = {33, 7, 163, 144, 145, 153};
        for (int idx : rightEyeIndices) {
            rightEye.push_back(face.landmarks[idx]);
        }
        
        if (leftEye.empty() || rightEye.empty()) {
            result.status = LivenessStatus::UNKNOWN;
            result.confidence = 0.0f;
            result.reason = "Could not extract eye landmarks";
            return result;
        }

        // 1. Blink detection
        float leftEAR = computeEAR(leftEye);
        float rightEAR = computeEAR(rightEye);
        
        if (leftEAR < 0.0f || rightEAR < 0.0f) {
            result.status = LivenessStatus::UNKNOWN;
            result.confidence = 0.0f;
            result.reason = "Could not compute EAR values";
            return result;
        }
        
        float avgEAR = (leftEAR + rightEAR) / 2.0f;
        Log::debug("LivenessChecker", "EAR: L=" + std::to_string(leftEAR) + 
                  " R=" + std::to_string(rightEAR) + " Avg=" + std::to_string(avgEAR));
        
        bool blinkDetected = detectBlink(avgEAR);
        
        // 2. Head pose detection
        float currentYaw = estimateHeadYaw(face.landmarks);
        float currentPitch = estimateHeadPitch(face.landmarks);
        Log::debug("LivenessChecker", "Head pose: Yaw=" + std::to_string(currentYaw) + 
                  " Pitch=" + std::to_string(currentPitch));
        
        bool headMovementDetected = detectHeadMovement(currentYaw, currentPitch);
        
        // 3. Decision logic - For video mode only
        auto now = std::chrono::steady_clock::now();
        double msSinceLastMove = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastHeadMoveTime_).count();

        Log::debug("LivenessChecker", "Time since last move: " + std::to_string(msSinceLastMove) + 
                  "ms, Frame count: " + std::to_string(frameCount_));

        if (blinkDetected) {
            result.status = LivenessStatus::LIVE;
            result.confidence = 0.95f;
            result.reason = "Blink detected";
            Log::info("LivenessChecker", "Blink detected!");
        } else if (headMovementDetected) {
            result.status = LivenessStatus::LIVE;
            result.confidence = 0.90f;
            result.reason = "Head movement detected";
            Log::info("LivenessChecker", "Head movement detected!");
        } else if (msSinceLastMove < config_.livenessWindowMs && isInitialized_) {
            result.status = LivenessStatus::LIVE;
            result.confidence = 0.75f;
            result.reason = "Recent movement within window";
        } else {
            result.status = LivenessStatus::NOT_LIVE;
            result.confidence = 0.80f;
            result.reason = "No liveness cues detected (no movement for " + 
                           std::to_string(msSinceLastMove) + "ms)";
        }
        
        // Update history for next frame
        lastYaw_ = currentYaw;
        lastPitch_ = currentPitch;
        
    } catch (const std::exception& e) {
        Log::error("LivenessChecker", "Liveness check error: " + std::string(e.what()));
        result.status = LivenessStatus::UNKNOWN;
        result.confidence = 0.0f;
        result.reason = "Processing error: " + std::string(e.what());
    }
    
    return result;
}