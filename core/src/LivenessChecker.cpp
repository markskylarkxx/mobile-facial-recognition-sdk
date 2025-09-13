//...................................................................
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
      isInitialized_(false),  // Add initialization flag
      frameCount_(0) {}       // Add frame counter

float LivenessChecker::computeEAR(const std::vector<Point>& eyeLandmarks) {
    // MediaPipe eye landmarks: 6 points per eye in this order:
    // [0] = outer corner, [1] = top-outer, [2] = top-inner, [3] = inner corner, [4] = bottom-inner, [5] = bottom-outer
    
    if (eyeLandmarks.size() != 6) {
        Log::warn("LivenessChecker", "Invalid eye landmarks count: " + std::to_string(eyeLandmarks.size()));
        return -1.0f; // Return invalid value instead of assuming open eyes
    }

    try {
        // Calculate EAR using the 6-point formula
        // EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        // Where P1-P6 are the 6 eye landmarks
        
        float vertical1 = std::sqrt(std::pow(eyeLandmarks[1].x - eyeLandmarks[5].x, 2) + 
                                   std::pow(eyeLandmarks[1].y - eyeLandmarks[5].y, 2));
        float vertical2 = std::sqrt(std::pow(eyeLandmarks[2].x - eyeLandmarks[4].x, 2) + 
                                   std::pow(eyeLandmarks[2].y - eyeLandmarks[4].y, 2));
        float horizontal = std::sqrt(std::pow(eyeLandmarks[0].x - eyeLandmarks[3].x, 2) + 
                                    std::pow(eyeLandmarks[0].y - eyeLandmarks[3].y, 2));
        
        if (horizontal < 1e-6f) {
            Log::warn("LivenessChecker", "Horizontal eye distance too small for EAR calculation");
            return -1.0f; // Return invalid value
        }
        
        float ear = (vertical1 + vertical2) / (2.0f * horizontal);
        
        // Clamp EAR to reasonable range
        return std::max(0.0f, std::min(1.0f, ear));
        
    } catch (const std::exception& e) {
        Log::error("LivenessChecker", "EAR calculation error: " + std::string(e.what()));
        return -1.0f; // Return invalid value
    }
}

float LivenessChecker::estimateHeadYaw(const std::vector<Point>& landmarks) {
    if (landmarks.size() != 468) {
        Log::warn("LivenessChecker", "Invalid landmarks count for MediaPipe: " + std::to_string(landmarks.size()));
        return 0.0f;
    }

    try {
        // Get nose tip
        const Point& noseTip = landmarks[1]; // MediaPipe nose tip index
        
        // Calculate eye centers from MediaPipe eye landmarks
        // Left eye landmarks (from viewer's perspective): 362, 382, 381, 380, 374, 373
        // Right eye landmarks: 33, 7, 163, 144, 145, 153
        
        // Left eye center (viewer's left, person's right)
        Point leftEyeCenter = {0.0f, 0.0f};
        std::vector<int> leftEyeIndices = {362, 382, 381, 380, 374, 373};
        for (int idx : leftEyeIndices) {
            leftEyeCenter.x += landmarks[idx].x;
            leftEyeCenter.y += landmarks[idx].y;
        }
        leftEyeCenter.x /= leftEyeIndices.size();
        leftEyeCenter.y /= leftEyeIndices.size();
        
        // Right eye center (viewer's right, person's left)
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
        
        // Calculate yaw based on nose position relative to eye center
        float eyesDistance = std::abs(rightEyeCenter.x - leftEyeCenter.x);
        
        if (eyesDistance < 1e-3f) {
            Log::warn("LivenessChecker", "Eyes too close for yaw calculation: " + std::to_string(eyesDistance));
            return 0.0f;
        }
        
        float delta = noseTip.x - eyesCenterX;
        float normalizedYaw = delta / eyesDistance;
        
        // Clamp to reasonable range (-1 to 1, roughly -45 to +45 degrees)
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
        // Use proper MediaPipe landmark indices for better pitch estimation
        const int NOSE_TIP = 1;      // Nose tip
        const int FOREHEAD = 10;     // Forehead point
        const int CHIN = 175;        // Chin point
        
        const Point& noseTip = landmarks[NOSE_TIP];
        const Point& forehead = landmarks[FOREHEAD];
        const Point& chin = landmarks[CHIN];
        
        // DEBUG: Print landmarks being used
        Log::debug("LivenessChecker", "Pitch landmarks:");
        Log::debug("LivenessChecker", "  Nose [1]: (" + std::to_string(noseTip.x) + ", " + std::to_string(noseTip.y) + ")");
        Log::debug("LivenessChecker", "  Forehead [10]: (" + std::to_string(forehead.x) + ", " + std::to_string(forehead.y) + ")");
        Log::debug("LivenessChecker", "  Chin [175]: (" + std::to_string(chin.x) + ", " + std::to_string(chin.y) + ")");
        
        // Calculate face height
        float faceHeight = std::abs(chin.y - forehead.y);
        if (faceHeight < 1e-3f) {
            Log::warn("LivenessChecker", "Face height too small for pitch calculation: " + std::to_string(faceHeight));
            return 0.0f;
        }
        
        // Calculate face center Y
        float faceCenterY = (forehead.y + chin.y) / 2.0f;
        
        // Normalize pitch estimate
        float normalizedPitch = (noseTip.y - faceCenterY) / faceHeight;
        
        // Clamp to reasonable range (-1 to 1)
        return std::max(-1.0f, std::min(1.0f, normalizedPitch));
        
    } catch (const std::exception& e) {
        Log::error("LivenessChecker", "Head pitch estimation error: " + std::string(e.what()));
        return 0.0f;
    }
}

bool LivenessChecker::detectBlink(float currentEAR) {
    // Skip if EAR is invalid
    if (currentEAR < 0.0f) {
        Log::warn("LivenessChecker", "Invalid EAR value, skipping blink detection");
        return false;
    }
    
    // Store current EAR in history
    earHistory_.push_back(currentEAR);
    if (earHistory_.size() > 10) { // Keep last 10 frames
        earHistory_.pop_front();
    }
    
    // Log EAR for debugging
    Log::debug("LivenessChecker", "Current EAR: " + std::to_string(currentEAR) + 
               ", Threshold: " + std::to_string(config_.earClosedThreshold) +
               ", Blink frames: " + std::to_string(blinkFrameCount_));
    
    // Check if current EAR indicates eye closure
    if (currentEAR < config_.earClosedThreshold) {
        blinkFrameCount_++;
        Log::debug("LivenessChecker", "Eyes closed, frame count: " + std::to_string(blinkFrameCount_));
        return false; // Eye is closed, but not yet a complete blink
    } else {
        // If eyes were closed for sufficient frames and now open, it's a blink
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
    // For the first frame, just initialize values
    if (!isInitialized_) {
        lastYaw_ = currentYaw;
        lastPitch_ = currentPitch;
        isInitialized_ = true;
        Log::info("LivenessChecker", "Initialized head pose tracking - Yaw: " + std::to_string(currentYaw) + 
                 ", Pitch: " + std::to_string(currentPitch));
        return false; // No movement on first frame
    }
    
    float yawChange = std::abs(currentYaw - lastYaw_);
    float pitchChange = std::abs(currentPitch - lastPitch_);
    
    // Convert to approximate degrees (empirical scaling)
    // Assuming normalized values (-1 to 1) correspond to roughly -45 to +45 degrees
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
    frameCount_++; // Increment frame counter
    
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
        // Extract eyes using MediaPipe indices directly
        // Left eye (viewer's perspective): 362, 382, 381, 380, 374, 373
        std::vector<Point> leftEye;
        std::vector<int> leftEyeIndices = {362, 382, 381, 380, 374, 373};
        for (int idx : leftEyeIndices) {
            leftEye.push_back(face.landmarks[idx]);
        }
        
        // Right eye (viewer's perspective): 33, 7, 163, 144, 145, 153  
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
        
        // Handle invalid EAR values
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
        
        // 3. Decision logic - FIXED to properly handle static images
        auto now = std::chrono::steady_clock::now();
        double msSinceLastMove = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastHeadMoveTime_).count();
        
        Log::debug("LivenessChecker", "Time since last move: " + std::to_string(msSinceLastMove) + 
                  "ms, Frame count: " + std::to_string(frameCount_));
        
        // CRITICAL FIX: For static images (single frame), always return NOT_LIVE
        if (frameCount_ == 1) {
            result.status = LivenessStatus::NOT_LIVE;
            result.confidence = 0.90f;
            result.reason = "Static image - no temporal data available";
            Log::info("LivenessChecker", "Static image detected - marked as NOT_LIVE");
        } else if (blinkDetected) {
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
            // Recent movement still within liveness window (but only if we've been tracking)
            result.status = LivenessStatus::LIVE;
            result.confidence = 0.75f;
            result.reason = "Recent movement within window";
        } else {
            // No recent movement detected
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


