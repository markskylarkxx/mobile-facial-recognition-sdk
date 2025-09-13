

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

namespace neptune {

class TfLiteEngine {
public:
    TfLiteEngine();
    ~TfLiteEngine();
  

    // Load a .tflite model from disk
    bool loadModel(const std::string& modelPath);

    // Optionally resize input tensor (NHWC). After calling, tensors are (re)allocated.
    bool resizeInputTensor(int width, int height, int channels);

    // Copy input data into the tensor (expects float32 NHWC, size == 1*H*W*C)
    bool setInputTensor(const std::vector<float>& inputData);

    // Run inference
    bool invoke();

    // Get output tensor data (float32). Returns empty vector on failure.
    std::vector<float> getOutputTensor(int index = 0) const;

    // Query input/output tensor info
    int getInputTensorSize() const;
    int getOutputTensorSize(int index = 0) const;
    int getNumOutputs() const;
    std::vector<int> getOutputTensorShape(int index) const;

    int inputWidth() const { return inputWidth_; }
    int inputHeight() const { return inputHeight_; }
    int inputChannels() const { return inputChannels_; }

    // Error reporting
    const std::string& getLastError() const { return lastError_; }
  
    // The new method to be added
    bool isLoaded() const;

private:
    void updateInputDims();

    std::unique_ptr<::tflite::FlatBufferModel> model_;
    std::unique_ptr<::tflite::Interpreter> interpreter_;

    int inputWidth_ = 0;
    int inputHeight_ = 0;
    int inputChannels_ = 0;

    std::string lastError_;
};


} // namespace neptune
