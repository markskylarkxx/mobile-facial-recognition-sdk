// File: core/src/MediaPipeLandmarks.cpp
#include "neptune/MediaPipeLandmarks.h"
#include "neptune/Log.h"
#include <cmath>
#include <algorithm>

namespace neptune {

std::vector<Point> MediaPipeLandmarks::extractEyeLandmarks(const std::vector<Point>& landmarks, bool leftEye) {
    std::vector<Point> eyePoints;
    
    const int* indices = leftEye ? LEFT_EYE_INDICES : RIGHT_EYE_INDICES;
    const int count = sizeof(LEFT_EYE_INDICES) / sizeof(LEFT_EYE_INDICES[0]);
    
    for (int i = 0; i < count; ++i) {
        int index = indices[i];
        if (index < static_cast<int>(landmarks.size())) {
            eyePoints.push_back(landmarks[index]);
        } else {
            Log::warn("MediaPipeLandmarks", 
                        "Landmark index " + std::to_string(index) + 
                        " out of bounds (total: " + std::to_string(landmarks.size()) + ")");
        }
    }
    
    return eyePoints;
}

Point MediaPipeLandmarks::calculateEyeCenter(const std::vector<Point>& eyeLandmarks) {
    if (eyeLandmarks.empty()) {
        Log::warn("MediaPipeLandmarks", "Empty eye landmarks for center calculation");
        return Point(0, 0);
    }
    
    Point center(0, 0);
    for (const auto& point : eyeLandmarks) {
        center.x += point.x;
        center.y += point.y;
    }
    
    center.x /= static_cast<float>(eyeLandmarks.size());
    center.y /= static_cast<float>(eyeLandmarks.size());
    
    return center;
}

float MediaPipeLandmarks::calculateEAR(const std::vector<Point>& eyeLandmarks) {
    if (eyeLandmarks.size() < 16) {
        Log::warn("MediaPipeLandmarks", 
                    "Insufficient eye landmarks for EAR calculation: " + 
                    std::to_string(eyeLandmarks.size()));
        return 1.0f;
    }
    
    auto dist = [](const Point& a, const Point& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    
    try {
        // MediaPipe 16-point eye model EAR calculation
        // Vertical distances (top to bottom of eye)
        float vertical1 = dist(eyeLandmarks[1], eyeLandmarks[7]);   // Top to bottom
        float vertical2 = dist(eyeLandmarks[2], eyeLandmarks[6]);   // Another vertical
        float vertical3 = dist(eyeLandmarks[3], eyeLandmarks[5]);   // Another vertical
        
        // Horizontal distance (eye width)
        float horizontal = dist(eyeLandmarks[0], eyeLandmarks[4]);
        
        if (horizontal < 1.0f) {
            Log::warn("MediaPipeLandmarks", "Horizontal eye distance too small: " + std::to_string(horizontal));
            return 1.0f;
        }
        
        return (vertical1 + vertical2 + vertical3) / (3.0f * horizontal);
        
    } catch (const std::exception& e) {
        Log::error("MediaPipeLandmarks", "EAR calculation error: " + std::string(e.what()));
        return 1.0f;
    }
}

bool MediaPipeLandmarks::validateLandmarks(const std::vector<Point>& landmarks, int expectedCount) {
    if (landmarks.empty()) {
        Log::warn("MediaPipeLandmarks", "No landmarks provided for validation");
        return false;
    }
    
    if (expectedCount > 0 && landmarks.size() != static_cast<size_t>(expectedCount)) {
        Log::warn("MediaPipeLandmarks", 
                    "Expected " + std::to_string(expectedCount) + 
                    " landmarks, got " + std::to_string(landmarks.size()));
        return false;
    }
    
    // Check for invalid points (NaN, infinity, or extreme values)
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const auto& point = landmarks[i];
        
        if (std::isnan(point.x) || std::isnan(point.y) ||
            std::isinf(point.x) || std::isinf(point.y)) {
            Log::warn("MediaPipeLandmarks", 
                        "Invalid landmark at index " + std::to_string(i) + 
                        ": (" + std::to_string(point.x) + ", " + std::to_string(point.y) + ")");
            return false;
        }
        
        // Check for reasonable coordinate values (assuming image coordinates)
        if (point.x < -1000 || point.x > 10000 || point.y < -1000 || point.y > 10000) {
            Log::warn("MediaPipeLandmarks", 
                        "Suspicious landmark value at index " + std::to_string(i) + 
                        ": (" + std::to_string(point.x) + ", " + std::to_string(point.y) + ")");
            return false;
        }
    }
    
    return true;
}
// Add these methods to MediaPipeLandmarks.cpp

std::vector<Point> MediaPipeLandmarks::extractLipLandmarks(const std::vector<Point>& landmarks) {
    std::vector<Point> lipPoints;
    
    const int count = sizeof(LIPS_INDICES) / sizeof(LIPS_INDICES[0]);
    
    for (int i = 0; i < count; ++i) {
        int index = LIPS_INDICES[i];
        if (index < static_cast<int>(landmarks.size())) {
            lipPoints.push_back(landmarks[index]);
        }
    }
    
    return lipPoints;
}

float MediaPipeLandmarks::calculateMAR(const std::vector<Point>& lipLandmarks) {
    if (lipLandmarks.size() < 20) {
        Log::warn("MediaPipeLandmarks", 
                    "Insufficient lip landmarks for MAR calculation: " + 
                    std::to_string(lipLandmarks.size()));
        return 0.0f;
    }
    
    auto dist = [](const Point& a, const Point& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    
    try {
        // Calculate vertical distances (mouth opening)
        float vertical1 = dist(lipLandmarks[2], lipLandmarks[10]);  // Upper to lower lip
        float vertical2 = dist(lipLandmarks[3], lipLandmarks[9]);   // Another vertical
        float vertical3 = dist(lipLandmarks[4], lipLandmarks[8]);   // Another vertical
        
        // Calculate horizontal distance (mouth width)
        float horizontal = dist(lipLandmarks[0], lipLandmarks[6]);
        
        if (horizontal < 1.0f) {
            return 0.0f;
        }
        
        return (vertical1 + vertical2 + vertical3) / (3.0f * horizontal);
        
    } catch (const std::exception& e) {
        Log::error("MediaPipeLandmarks", "MAR calculation error: " + std::string(e.what()));
        return 0.0f;
    }
}
} // namespace neptune