
#include "neptune/LivenessChecker.h"
#include "neptune/MediaPipeLandmarks.h"
#include "neptune/Log.h"
#include <cmath>
#include <algorithm>
#include <chrono>
using namespace neptune;

LivenessChecker::LivenessChecker(const NeptuneConfig& config)
    : config_(config),
      lastHeadMoveTime_(std::chrono::steady_clock::now()),
      firstDetectionTime_(std::chrono::steady_clock::now()),
      blinkFrameCount_(0),
      lastYaw_(0.0f),
      lastPitch_(0.0f),
      smoothedYaw_(0.0f),
      smoothedPitch_(0.0f),
      isInitialized_(false),
      frameCount_(0),
      isVideoMode_(false),
      hasProvenLiveness_(false),
      totalBlinksDetected_(0),
      totalHeadMovements_(0),
      baselineEAR_(0.3f), // Initial guess
      calibrationFrames_(0) {}

void LivenessChecker::setVideoMode(bool enabled) {
    isVideoMode_ = enabled;
    Log::info("LivenessChecker", "Video mode set to: " + std::string(enabled ? "true" : "false"));
    if (!enabled) {
        resetForNewFrame();
    }
}

void LivenessChecker::resetForNewFrame() {
    frameCount_ = 0;
    isInitialized_ = false;
    earHistory_.clear();
    yawHistory_.clear();
    pitchHistory_.clear();
    blinkFrameCount_ = 0;
    lastHeadMoveTime_ = std::chrono::steady_clock::now();
    firstDetectionTime_ = std::chrono::steady_clock::now();
    lastYaw_ = 0.0f;
    lastPitch_ = 0.0f;
    smoothedYaw_ = 0.0f;
    smoothedPitch_ = 0.0f;
    hasProvenLiveness_ = false;
    totalBlinksDetected_ = 0;
    totalHeadMovements_ = 0;
    baselineEAR_ = 0.3f;
    calibrationFrames_ = 0;
    Log::debug("LivenessChecker", "Reset for new frame/image - liveness proof required");
}

float LivenessChecker::computeEAR(const std::vector<Point>& eyeLandmarks) {
    if (eyeLandmarks.size() != 6) {
        Log::warn("LivenessChecker", "Invalid eye landmarks count: " + std::to_string(eyeLandmarks.size()));
        return -1.0f;
    }
    try {
        Log::debug("LivenessChecker", "Eye landmarks: ");
        for (int i = 0; i < 6; i++) {
            Log::debug("LivenessChecker", " P" + std::to_string(i) + ": (" +
                      std::to_string(eyeLandmarks[i].x) + ", " + std::to_string(eyeLandmarks[i].y) + ")");
        }
        // EAR formula: (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        float vertical1 = std::sqrt(std::pow(eyeLandmarks[1].x - eyeLandmarks[5].x, 2) +
                                   std::pow(eyeLandmarks[1].y - eyeLandmarks[5].y, 2));
        float vertical2 = std::sqrt(std::pow(eyeLandmarks[2].x - eyeLandmarks[4].x, 2) +
                                   std::pow(eyeLandmarks[2].y - eyeLandmarks[4].y, 2));
        float horizontal = std::sqrt(std::pow(eyeLandmarks[0].x - eyeLandmarks[3].x, 2) +
                                    std::pow(eyeLandmarks[0].y - eyeLandmarks[3].y, 2));
        Log::debug("LivenessChecker", "EAR components: vertical1=" + std::to_string(vertical1) +
                  ", vertical2=" + std::to_string(vertical2) + ", horizontal=" + std::to_string(horizontal));
        if (horizontal < 1e-6f) {
            Log::warn("LivenessChecker", "Horizontal eye distance too small: " + std::to_string(horizontal));
            return -1.0f;
        }
        float ear = (vertical1 + vertical2) / (2.0f * horizontal);
        Log::debug("LivenessChecker", "Computed EAR: " + std::to_string(ear));
        return ear;
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
        const Point& noseTip = landmarks[1];
        Point leftEyeCenter = {0.0f, 0.0f};
        std::vector<int> leftEyeIndices = {362, 385, 387, 263, 373, 380};
        for (int idx : leftEyeIndices) {
            leftEyeCenter.x += landmarks[idx].x;
            leftEyeCenter.y += landmarks[idx].y;
        }
        leftEyeCenter.x /= leftEyeIndices.size();
        leftEyeCenter.y /= leftEyeIndices.size();
        Point rightEyeCenter = {0.0f, 0.0f};
        std::vector<int> rightEyeIndices = {33, 159, 158, 133, 145, 153};
        for (int idx : rightEyeIndices) {
            rightEyeCenter.x += landmarks[idx].x;
            rightEyeCenter.y += landmarks[idx].y;
        }
        rightEyeCenter.x /= rightEyeIndices.size();
        rightEyeCenter.y /= rightEyeIndices.size();
        Log::debug("LivenessChecker", "Left eye center: (" + std::to_string(leftEyeCenter.x) + ", " + std::to_string(leftEyeCenter.y) + ")");
        Log::debug("LivenessChecker", "Right eye center: (" + std::to_string(rightEyeCenter.x) + ", " + std::to_string(rightEyeCenter.y) + ")");
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
        Log::debug("LivenessChecker", " Nose [1]: (" + std::to_string(noseTip.x) + ", " + std::to_string(noseTip.y) + ")");
        Log::debug("LivenessChecker", " Forehead [10]: (" + std::to_string(forehead.x) + ", " + std::to_string(forehead.y) + ")");
        Log::debug("LivenessChecker", " Chin [175]: (" + std::to_string(chin.x) + ", " + std::to_string(chin.y) + ")");
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
    if (earHistory_.size() > 15) {
        earHistory_.pop_front();
    }
    Log::debug("LivenessChecker", "Current EAR: " + std::to_string(currentEAR));
    // Calibration phase: compute baseline EAR over first 10 frames
    if (calibrationFrames_ < 10) {
        baselineEAR_ = (baselineEAR_ * calibrationFrames_ + currentEAR) / (calibrationFrames_ + 1);
        calibrationFrames_++;
        Log::debug("LivenessChecker", "Calibrating baseline EAR: " + std::to_string(baselineEAR_) +
                  ", frame " + std::to_string(calibrationFrames_));
        return false; // No blink detection during calibration
    }
    float adaptiveThreshold = baselineEAR_ * 0.6f; // Adjusted multiplier
    adaptiveThreshold = std::max(0.12f, std::min(adaptiveThreshold, 0.25f)); // Wider range
    if (earHistory_.size() >= 5) {
        float avgEAR = 0.0f;
        float minEAR = std::numeric_limits<float>::max();
        float maxEAR = std::numeric_limits<float>::min();
        for (float ear : earHistory_) {
            avgEAR += ear;
            minEAR = std::min(minEAR, ear);
            maxEAR = std::max(maxEAR, ear);
        }
        avgEAR /= earHistory_.size();
        Log::debug("LivenessChecker", "EAR stats: avg=" + std::to_string(avgEAR) +
                  ", min=" + std::to_string(minEAR) + ", max=" + std::to_string(maxEAR) +
                  ", baseline=" + std::to_string(baselineEAR_) +
                  ", threshold=" + std::to_string(adaptiveThreshold) +
                  ", blink_frames=" + std::to_string(blinkFrameCount_));
    }
    if (currentEAR < adaptiveThreshold) {
        blinkFrameCount_++;
        Log::debug("LivenessChecker", "Eyes closing/closed, frame count: " + std::to_string(blinkFrameCount_));
        return false;
    } else {
        if (blinkFrameCount_ >= config_.blinkMinFrames && blinkFrameCount_ <= 8) {
            totalBlinksDetected_++;
            Log::info("LivenessChecker", "BLINK DETECTED! Closed for " + std::to_string(blinkFrameCount_) +
                     " frames. EAR dropped to " + std::to_string(currentEAR) + " (threshold: " + std::to_string(adaptiveThreshold) +
                     "). Total blinks: " + std::to_string(totalBlinksDetected_));
            blinkFrameCount_ = 0;
            return true;
        } else if (blinkFrameCount_ > 8) {
            Log::debug("LivenessChecker", "Too many closed frames (" + std::to_string(blinkFrameCount_) +
                      ") - sustained eye closure, not a blink");
        }
        blinkFrameCount_ = 0;
        return false;
    }
}

bool LivenessChecker::detectHeadMovement(float currentYaw, float currentPitch) {
    yawHistory_.push_back(currentYaw);
    pitchHistory_.push_back(currentPitch);
    const size_t MAX_HISTORY = 15;
    if (yawHistory_.size() > MAX_HISTORY) {
        yawHistory_.pop_front();
    }
    if (pitchHistory_.size() > MAX_HISTORY) {
        pitchHistory_.pop_front();
    }
    if (!isInitialized_) {
        smoothedYaw_ = currentYaw;
        smoothedPitch_ = currentPitch;
        lastYaw_ = currentYaw;
        lastPitch_ = currentPitch;
        isInitialized_ = true;
        Log::info("LivenessChecker", "Initialized head pose tracking - Yaw: " + std::to_string(currentYaw) +
                 ", Pitch: " + std::to_string(currentPitch));
        return false;
    }
    const float alpha = 0.15f; // Reduced for more responsiveness
    smoothedYaw_ = alpha * currentYaw + (1.0f - alpha) * smoothedYaw_;
    smoothedPitch_ = alpha * currentPitch + (1.0f - alpha) * smoothedPitch_;
    float yawChange = std::abs(smoothedYaw_ - lastYaw_) * 45.0f;
    float pitchChange = std::abs(smoothedPitch_ - lastPitch_) * 45.0f;
    Log::debug("LivenessChecker", "Instant changes: Yaw=" + std::to_string(yawChange) +
              "°, Pitch=" + std::to_string(pitchChange) + "°");
    const float YAW_THRESHOLD = 2.0f; // Lowered for sensitivity
    const float PITCH_THRESHOLD = 1.5f; // Lowered for sensitivity
    bool movementDetected = (yawChange > YAW_THRESHOLD) || (pitchChange > PITCH_THRESHOLD);
    if (movementDetected) {
        auto now = std::chrono::steady_clock::now();
        double msSinceLast = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastHeadMoveTime_).count();
        if (msSinceLast < 500.0) { // Increased debounce interval
            movementDetected = false;
            Log::debug("LivenessChecker", "Head movement ignored due to debounce");
        } else {
            totalHeadMovements_++;
            lastHeadMoveTime_ = now;
            Log::info("LivenessChecker", "HEAD MOVEMENT DETECTED: Yaw=" + std::to_string(yawChange) +
                     "°, Pitch=" + std::to_string(pitchChange) + "°. Total: " + std::to_string(totalHeadMovements_));
        }
    }
    lastYaw_ = smoothedYaw_;
    lastPitch_ = smoothedPitch_;
    return movementDetected;
}

LivenessResult LivenessChecker::check(const FaceBox& face) {
    LivenessResult result;
    frameCount_++;
    if (!isVideoMode_) {
        result.status = LivenessStatus::NOT_LIVE;
        result.confidence = 0.95f;
        result.reason = "Static image - no temporal data available";
        Log::info("LivenessChecker", "Static image detected - marked as NOT_LIVE");
        return result;
    }
    if (face.landmarks.empty()) {
        result.status = LivenessStatus::NOT_LIVE;
        result.confidence = 0.8f;
        result.reason = "No landmarks available - cannot verify liveness";
        return result;
    }
    if (face.landmarks.size() != 468) {
        result.status = LivenessStatus::NOT_LIVE;
        result.confidence = 0.8f;
        result.reason = "Invalid landmark count: " + std::to_string(face.landmarks.size()) + " (expected 468)";
        Log::error("LivenessChecker", result.reason);
        return result;
    }
    try {
        std::vector<Point> leftEye;
        std::vector<int> leftEyeIndices = {362, 385, 387, 263, 373, 380};
        for (int idx : leftEyeIndices) {
            if (idx < face.landmarks.size()) {
                leftEye.push_back(face.landmarks[idx]);
            }
        }
        std::vector<Point> rightEye;
        std::vector<int> rightEyeIndices = {33, 159, 158, 133, 145, 153};
        for (int idx : rightEyeIndices) {
            if (idx < face.landmarks.size()) {
                rightEye.push_back(face.landmarks[idx]);
            }
        }
        if (leftEye.size() != 6 || rightEye.size() != 6) {
            result.status = LivenessStatus::NOT_LIVE;
            result.confidence = 0.8f;
            result.reason = "Could not extract eye landmarks - cannot verify liveness";
            return result;
        }
        float leftEAR = computeEAR(leftEye);
        float rightEAR = computeEAR(rightEye);
        if (leftEAR < 0.0f || rightEAR < 0.0f) {
            result.status = LivenessStatus::NOT_LIVE;
            result.confidence = 0.8f;
            result.reason = "Could not compute EAR values - cannot verify liveness";
            return result;
        }
        float avgEAR = (leftEAR + rightEAR) / 2.0f;
        Log::debug("LivenessChecker", "EAR: L=" + std::to_string(leftEAR) +
                  " R=" + std::to_string(rightEAR) + " Avg=" + std::to_string(avgEAR));
        bool blinkDetected = detectBlink(avgEAR);
        float currentYaw = estimateHeadYaw(face.landmarks);
        float currentPitch = estimateHeadPitch(face.landmarks);
        Log::debug("LivenessChecker", "Head pose: Yaw=" + std::to_string(currentYaw) +
                  " Pitch=" + std::to_string(currentPitch));
        bool headMovementDetected = detectHeadMovement(currentYaw, currentPitch);
        auto now = std::chrono::steady_clock::now();
        double msSinceFirstDetection = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - firstDetectionTime_).count();
        double msSinceLastMove = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastHeadMoveTime_).count();
        Log::debug("LivenessChecker", "Time since first detection: " + std::to_string(msSinceFirstDetection) +
                  "ms, Time since last move: " + std::to_string(msSinceLastMove) +
                  "ms, Frame count: " + std::to_string(frameCount_));
        // Relaxed liveness condition: 1 blink + 1 head movement OR 2 blinks OR 2 head movements
        if ((totalBlinksDetected_ >= 1 && totalHeadMovements_ >= 1) ||
            totalBlinksDetected_ >= 2 || totalHeadMovements_ >= 2) {
            if (!hasProvenLiveness_) {
                hasProvenLiveness_ = true;
                Log::info("LivenessChecker", "LIVENESS PROVEN! Blinks: " + std::to_string(totalBlinksDetected_) +
                         ", Head movements: " + std::to_string(totalHeadMovements_));
            }
        }
        if (!hasProvenLiveness_) {
            const double PROBATION_PERIOD_MS = 20000.0; // Extended to 20 seconds
            if (msSinceFirstDetection < PROBATION_PERIOD_MS) {
                result.status = LivenessStatus::NOT_LIVE;
                result.confidence = 0.60f;
                result.reason = "Awaiting liveness proof (" +
                               std::to_string(static_cast<int>((PROBATION_PERIOD_MS - msSinceFirstDetection) / 1000)) +
                               "s remaining) - please blink and move your head";
                Log::debug("LivenessChecker", "Still in probation period, awaiting liveness proof");
            } else {
                result.status = LivenessStatus::NOT_LIVE;
                result.confidence = 0.90f;
                result.reason = "No liveness detected - likely a photo or static image";
                Log::warn("LivenessChecker", "Probation period expired without liveness proof - marking as NOT_LIVE");
            }
        } else {
            if (msSinceLastMove < config_.livenessWindowMs) {
                result.status = LivenessStatus::LIVE;
                result.confidence = 0.85f + (totalBlinksDetected_ * 0.02f) + (totalHeadMovements_ * 0.03f);
                result.confidence = std::min(0.98f, result.confidence);
                result.reason = "Liveness confirmed (blinks: " + std::to_string(totalBlinksDetected_) +
                               ", movements: " + std::to_string(totalHeadMovements_) + ")";
            } else {
                result.status = LivenessStatus::NOT_LIVE;
                result.confidence = 0.75f;
                result.reason = "Liveness expired - no recent movement for " +
                               std::to_string(msSinceLastMove) + "ms (had proven liveness before)";
                if (msSinceLastMove > config_.livenessWindowMs * 2) {
                    Log::info("LivenessChecker", "Resetting liveness proof due to extended inactivity");
                    hasProvenLiveness_ = false;
                    totalBlinksDetected_ = 0;
                    totalHeadMovements_ = 0;
                    firstDetectionTime_ = now;
                }
            }
        }
    } catch (const std::exception& e) {
        Log::error("LivenessChecker", "Liveness check error: " + std::string(e.what()));
        result.status = LivenessStatus::NOT_LIVE;
        result.confidence = 0.8f;
        result.reason = "Processing error - cannot verify liveness: " + std::string(e.what());
    }
    return result;
}