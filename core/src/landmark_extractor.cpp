
// #include "neptune/landmark_extractor.h"
// #include "neptune/Log.h"

// #include <tensorflow/lite/kernels/register.h>
// #include <tensorflow/lite/model.h>

// LandmarkExtractor::LandmarkExtractor(const std::string& modelPath) {
//     model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//     if (!model) {
//      //   LOG_ERROR("Failed to load landmark model: " + modelPath);
//         return;
//     }

//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder(*model, resolver)(&interpreter);

//     if (!interpreter) {
//       //  LOG_ERROR("Failed to create interpreter for landmark model");
//         return;
//     }

//     interpreter->AllocateTensors();

//     // Model input dims (N,H,W,C)
//     auto* inputTensor = interpreter->tensor(interpreter->inputs()[0]);
//     inputHeight = inputTensor->dims->data[1];
//     inputWidth  = inputTensor->dims->data[2];

//     //LOG_INFO("[LandmarkExtractor] Model expects input: "
//           //   + std::to_string(inputWidth) + "x" + std::to_string(inputHeight));
// }

// std::vector<neptune::Point> LandmarkExtractor::Process(const cv::Mat& image, const cv::Rect& faceRect) {
//     std::vector<neptune::Point> landmarks;
//     if (!interpreter) return landmarks;

//     // Crop face ROI
//     cv::Rect roi = faceRect & cv::Rect(0, 0, image.cols, image.rows);
//     if (roi.width <= 0 || roi.height <= 0) return landmarks;
    
//     cv::Mat faceROI = image(roi).clone();
//     cv::Mat resized;
//     cv::resize(faceROI, resized, cv::Size(inputWidth, inputHeight));

//     // MediaPipe landmark models expect [0, 1] normalized input
//     resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

//     // Copy into tensor (ensure proper channel ordering)
//     float* input = interpreter->typed_input_tensor<float>(0);

//     // Convert BGR to RGB and copy with proper layout
//     cv::Mat rgb;
//     cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

//     // Copy data with proper memory layout (HWC format)
//     memcpy(input, rgb.data, sizeof(float) * inputWidth * inputHeight * 3);

//     // Run inference
//     if (interpreter->Invoke() != kTfLiteOk) {
//         return landmarks;
//     }

//     // Extract raw landmarks
//     auto* outputTensor = interpreter->tensor(interpreter->outputs()[0]);
//     int numValues = outputTensor->bytes / sizeof(float);
//     const float* output = interpreter->typed_output_tensor<float>(0);
//     int numLandmarks = numValues / 3; // (x,y,z)

//     landmarks.reserve(numLandmarks);

//     for (int i = 0; i < numLandmarks; i++) {
//         // MediaPipe outputs coordinates in [-1, 1] range
//         float rawX = output[i * 3 + 0];
//         float rawY = output[i * 3 + 1];
        
//         // Convert [-1,1] to [0,1] normalized coordinates
//         float normalizedX = (rawX + 1.0f) * 0.5f;
//         float normalizedY = (rawY + 1.0f) * 0.5f;
        
//         // FIXED: Map to face ROI coordinates first, then to absolute image coordinates
//         // Keep as float until final conversion to avoid precision loss
//         float roiX = normalizedX * static_cast<float>(roi.width);
//         float roiY = normalizedY * static_cast<float>(roi.height);
        
//         // Convert to absolute image coordinates
//         float absX = roiX + static_cast<float>(roi.x);
//         float absY = roiY + static_cast<float>(roi.y);
        
//         // Clamp to image boundaries for safety
//         absX = std::max(0.0f, std::min(absX, static_cast<float>(image.cols - 1)));
//         absY = std::max(0.0f, std::min(absY, static_cast<float>(image.rows - 1)));
        
//         landmarks.push_back({absX, absY});
//     }

//     // Enhanced debug output
//     if (!landmarks.empty()) {
//         std::cout << "DEBUG: Image size: " << image.cols << "x" << image.rows << "\n";
//         std::cout << "DEBUG: ROI: (" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << ")\n";
//         std::cout << "DEBUG: Model input size: " << inputWidth << "x" << inputHeight << "\n";
        
//         // Show first few landmarks for debugging
//         for (int j = 0; j < std::min(3, static_cast<int>(landmarks.size())); j++) {
//             float rawX = output[j * 3 + 0];
//             float rawY = output[j * 3 + 1];
//             float normX = rawX / static_cast<float>(inputWidth);
//             float normY = rawY / static_cast<float>(inputHeight);
//             std::cout << "DEBUG: Landmark " << j << " - raw: (" << rawX << ", " << rawY 
//                       << "), normalized: (" << normX << ", " << normY << ")"
//                       << "), absolute: (" << landmarks[j].x << ", " << landmarks[j].y << ")\n";
//         }

//         // Check if coordinates are reasonable
//         bool allValid = true;
//         for (const auto& landmark : landmarks) {
//             if (landmark.x < 0 || landmark.x >= image.cols ||
//                 landmark.y < 0 || landmark.y >= image.rows) {
//                 allValid = false;
//                 break;
//             }
//         }
        
//         if (!allValid) {
//             std::cout << "WARNING: Some landmark coordinates are out of bounds!\n";
//             std::cout << "  Expected range: [0, " << image.cols << ") x [0, " << image.rows << ")\n";
//         } else {
//             std::cout << "SUCCESS: All landmark coordinates are within bounds!\n";
//         }
//     }

//     return landmarks;
// }














/////////////////////////////////////////////////////////////////////////////////////////

#include "neptune/landmark_extractor.h"
#include "neptune/Log.h"

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

LandmarkExtractor::LandmarkExtractor(const std::string& modelPath) {
    model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model) {
        // LOG_ERROR("Failed to load landmark model: " + modelPath);
        return;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter) {
        // LOG_ERROR("Failed to create interpreter for landmark model");
        return;
    }

    interpreter->AllocateTensors();

    // Model input dims (N,H,W,C)
    auto* inputTensor = interpreter->tensor(interpreter->inputs()[0]);
    inputHeight = inputTensor->dims->data[1];
    inputWidth = inputTensor->dims->data[2];

    // LOG_INFO("[LandmarkExtractor] Model expects input: "
    //       + std::to_string(inputWidth) + "x" + std::to_string(inputHeight));
}

std::vector<neptune::Point> LandmarkExtractor::Process(const cv::Mat& image, const cv::Rect& faceRect) {
    std::vector<neptune::Point> landmarks;
    if (!interpreter) return landmarks;

    // Crop face ROI
    cv::Rect roi = faceRect & cv::Rect(0, 0, image.cols, image.rows);
    if (roi.width <= 0 || roi.height <= 0) return landmarks;
    
    cv::Mat faceROI = image(roi).clone();
    cv::Mat resized;
    cv::resize(faceROI, resized, cv::Size(inputWidth, inputHeight));

    // MediaPipe landmark models expect [0, 1] normalized input
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // Copy into tensor (ensure proper channel ordering)
    float* input = interpreter->typed_input_tensor<float>(0);

    // Convert BGR to RGB and copy with proper layout
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Copy data with proper memory layout (HWC format)
    memcpy(input, rgb.data, sizeof(float) * inputWidth * inputHeight * 3);

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        return landmarks;
    }

    // Extract raw landmarks
    auto* outputTensor = interpreter->tensor(interpreter->outputs()[0]);
    int numValues = outputTensor->bytes / sizeof(float);
    const float* output = interpreter->typed_output_tensor<float>(0);
    int numLandmarks = numValues / 3; // (x,y,z)

    landmarks.reserve(numLandmarks);

    for (int i = 0; i < numLandmarks; i++) {
        // MediaPipe outputs coordinates in [0, inputWidth/inputHeight] range (e.g., [0, 192])
        float rawX = output[i * 3 + 0];
        float rawY = output[i * 3 + 1];
        
        // Normalize to [0,1] relative to input dimensions
        float normalizedX = rawX / static_cast<float>(inputWidth);
        float normalizedY = rawY / static_cast<float>(inputHeight);
        
        // Map to face ROI coordinates, then to absolute image coordinates
        float roiX = normalizedX * static_cast<float>(roi.width);
        float roiY = normalizedY * static_cast<float>(roi.height);
        
        // Convert to absolute image coordinates
        float absX = roiX + static_cast<float>(roi.x);
        float absY = roiY + static_cast<float>(roi.y);
        
        // Clamp to image boundaries for safety
        absX = std::max(0.0f, std::min(absX, static_cast<float>(image.cols - 1)));
        absY = std::max(0.0f, std::min(absY, static_cast<float>(image.rows - 1)));
        
        landmarks.push_back({absX, absY});
    }

    // Enhanced debug output
    if (!landmarks.empty()) {
        std::cout << "DEBUG: Image size: " << image.cols << "x" << image.rows << "\n";
        std::cout << "DEBUG: ROI: (" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << ")\n";
        std::cout << "DEBUG: Model input size: " << inputWidth << "x" << inputHeight << "\n";
        
        // Show first few landmarks for debugging
        for (int j = 0; j < std::min(3, static_cast<int>(landmarks.size())); j++) {
            float rawX = output[j * 3 + 0];
            float rawY = output[j * 3 + 1];
            float normX = rawX / static_cast<float>(inputWidth);
            float normY = rawY / static_cast<float>(inputHeight);
            std::cout << "DEBUG: Landmark " << j << " - raw: (" << rawX << ", " << rawY 
                      << "), normalized: (" << normX << ", " << normY << ")"
                      << "), absolute: (" << landmarks[j].x << ", " << landmarks[j].y << ")\n";
        }

        // Check if coordinates are reasonable
        bool allValid = true;
        for (const auto& landmark : landmarks) {
            if (landmark.x < 0 || landmark.x >= image.cols ||
                landmark.y < 0 || landmark.y >= image.rows) {
                allValid = false;
                break;
            }
        }
        
        if (!allValid) {
            std::cout << "WARNING: Some landmark coordinates are out of bounds!\n";
            std::cout << " Expected range: [0, " << image.cols << ") x [0, " << image.rows << ")\n";
        } else {
            std::cout << "SUCCESS: All landmark coordinates are within bounds!\n";
        }
    }

    return landmarks;
}