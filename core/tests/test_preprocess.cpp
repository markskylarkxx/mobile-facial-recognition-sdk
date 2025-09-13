// NeptuneFacialSDK/core/tests/test_preprocess.cpp
#include <opencv2/opencv.hpp>
#include "neptune/Preprocess.h"
#include "neptune/Log.h"

int main() {
    cv::Mat img = cv::imread("../tests/assets/new_face.jpeg");  
    if (img.empty()) {
        neptune::Log::error("Test", "Could not load test.jpg");
        return -1;
    }

    // Step 1: Resize
    cv::Mat resized = neptune::img::Preprocess::resize(img, 224, 224);
    neptune::Log::info("Test", "Resized image to 224x224");

    // Step 2: Normalize
    std::vector<float> data = neptune::img::Preprocess::normalize(resized);
    neptune::Log::info("Test", "Normalized image. Vector size: " + std::to_string(data.size()));

    return 0;
}
