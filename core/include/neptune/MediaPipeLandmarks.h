// File: core/include/neptune/MediaPipeLandmarks.h
#pragma once

#include "neptune/Types.h"
#include <vector>

namespace neptune {

class MediaPipeLandmarks {
public:
    // MediaPipe face landmark indices (0-based for 468-point model)
    
    // Face contour/oval indices
    static constexpr int FACE_OVAL_INDICES[36] = {
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    };
    
    // Left eye indices (16 points)
    static constexpr int LEFT_EYE_INDICES[16] = {
        33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153,
        145, 144, 163, 7
    };
    
    // Right eye indices (16 points)
    static constexpr int RIGHT_EYE_INDICES[16] = {
        362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373,
        374, 380, 381, 382
    };
    
    // Lips/ mouth indices
    static constexpr int LIPS_INDICES[20] = {
        61, 146, 91, 181, 84, 17, 314, 405, 320, 307,
        325, 308, 78, 191, 80, 81, 82, 13, 312, 311
    };
    
    // Key facial feature points
    static constexpr int NOSE_TIP_INDEX = 4;
    static constexpr int FOREHEAD_INDEX = 10;
    static constexpr int CHIN_INDEX = 152;
    static constexpr int LEFT_EYE_INNER_CORNER = 33;
    static constexpr int LEFT_EYE_OUTER_CORNER = 133;
    static constexpr int RIGHT_EYE_INNER_CORNER = 362;
    static constexpr int RIGHT_EYE_OUTER_CORNER = 263;
    static constexpr int MOUTH_LEFT_CORNER = 61;
    static constexpr int MOUTH_RIGHT_CORNER = 291;

    /**
     * @brief Extract eye landmarks from full face landmarks
     * @param landmarks Full set of facial landmarks
     * @param leftEye true for left eye, false for right eye
     * @return Vector of eye landmarks
     */
    static std::vector<Point> extractEyeLandmarks(const std::vector<Point>& landmarks, bool leftEye);
    
    /**
     * @brief Calculate the center point of an eye
     * @param eyeLandmarks Landmarks for a single eye
     * @return Center point of the eye
     */
    static Point calculateEyeCenter(const std::vector<Point>& eyeLandmarks);
    
    /**
     * @brief Calculate Eye Aspect Ratio (EAR) for blink detection
     * @param eyeLandmarks Landmarks for a single eye
     * @return EAR value (lower values indicate eye closure)
     */
    static float calculateEAR(const std::vector<Point>& eyeLandmarks);
    
    /**
     * @brief Validate landmark points for correctness
     * @param landmarks Landmarks to validate
     * @param expectedCount Expected number of landmarks (0 for any)
     * @return true if landmarks are valid, false otherwise
     */
    static bool validateLandmarks(const std::vector<Point>& landmarks, int expectedCount = 0);
    
    /**
     * @brief Extract lip/mouth landmarks
     * @param landmarks Full set of facial landmarks
     * @return Vector of lip landmarks
     */
    static std::vector<Point> extractLipLandmarks(const std::vector<Point>& landmarks);
    
    /**
     * @brief Calculate mouth aspect ratio for mouth opening detection
     * @param lipLandmarks Lip landmarks
     * @return MAR value
     */
    static float calculateMAR(const std::vector<Point>& lipLandmarks);
};

} // namespace neptune