//
// File: NeptuneFacialSDK/core/src/img/Preprocess.h
//
// This file declares the Preprocess utility class.
// It contains static methods for all image preprocessing steps
// required before feeding an image into a TensorFlow Lite model.
//

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // 👈 needed for cv::resize

#include <vector>

namespace neptune {
    namespace img {
    
    class Preprocess {
    public:
        // Resize and optionally pad image to model input size
        static cv::Mat resize(const cv::Mat& img, int width, int height);
    
        // Convert BGR->RGB, normalize to 0-1, and flatten to NHWC vector
        static std::vector<float> normalize(const cv::Mat& img);
    };
    
    } // namespace img
    } // namespace neptune