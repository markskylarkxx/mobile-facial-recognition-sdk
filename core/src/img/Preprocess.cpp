//
// File: NeptuneFacialSDK/core/src/img/Preprocess.cpp
//
// This file implements the Preprocess utility class, using OpenCV to perform
// image manipulation tasks required for our machine learning models.
//

#include "neptune/Preprocess.h"
#include "neptune/Log.h"

using neptune::Log;  

namespace neptune {
    namespace img {
    
        cv::Mat Preprocess::resize(const cv::Mat& img, int targetWidth, int targetHeight) {
            int originalWidth = img.cols;
            int originalHeight = img.rows;
        
            float scale = std::min(targetWidth / float(originalWidth), targetHeight / float(originalHeight));
            int newWidth = int(originalWidth * scale);
            int newHeight = int(originalHeight * scale);
        
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(newWidth, newHeight));
        
            // Create a black canvas of target size
            cv::Mat output = cv::Mat::zeros(targetHeight, targetWidth, img.type());
        
            // Center the resized image
            int xOffset = (targetWidth - newWidth) / 2;
            int yOffset = (targetHeight - newHeight) / 2;
            resized.copyTo(output(cv::Rect(xOffset, yOffset, newWidth, newHeight)));
        
            Log::info("Preprocess", "Resized with padding to " + std::to_string(targetWidth) + "x" + std::to_string(targetHeight));
        
            return output;
        }
        
    
    std::vector<float> Preprocess::normalize(const cv::Mat& img) {
        cv::Mat rgbImg;
        // Convert BGR (OpenCV default) to RGB
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
    
        // Convert to float and scale to 0-1
        cv::Mat floatImg;
        rgbImg.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);
    
        // Flatten in NHWC order
        std::vector<float> processedData;
        processedData.reserve(floatImg.total() * floatImg.channels());
    
        for (int i = 0; i < floatImg.rows; ++i) {
            for (int j = 0; j < floatImg.cols; ++j) {
                cv::Vec3f pixel = floatImg.at<cv::Vec3f>(i, j);
                processedData.push_back(pixel[0]); // R
                processedData.push_back(pixel[1]); // G
                processedData.push_back(pixel[2]); // B
            }
        }
    
        Log::info("Preprocess", "Normalized image. Vector size: " + std::to_string(processedData.size()));
        return processedData;
    }
    
    } // namespace img
    } // namespace neptune




