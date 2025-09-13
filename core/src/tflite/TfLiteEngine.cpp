
// // This file implements the TfLiteEngine class, providing a robust interface
// // for loading and executing TensorFlow Lite models. i.e tflite models..
// //is your generic TensorFlow Lite runtime wrapper.
// // It loads .tflite models, allocates tensors, and runs inference.

// // File: NeptuneFacialSDK/core/src/tflite/TfLiteEngine.cpp

// #include "neptune/TfLiteEngine.h"
// #include "neptune/Log.h"
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"

// using neptune::Log;

// namespace tflite {

// TfLiteEngine::TfLiteEngine() : model_(nullptr), interpreter_(nullptr), inputWidth_(0), inputHeight_(0), inputChannels_(0) {}
// TfLiteEngine::~TfLiteEngine() = default;

// bool TfLiteEngine::loadModel(const std::string& modelPath) {
//     model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//     if (!model_) {
//         Log::error("TfLiteEngine", "Failed to load TFLite model from path: " + modelPath);
//         return false;
//     }

//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
//     if (!interpreter_) {
//         Log::error("TfLiteEngine", "Failed to create interpreter.");
//         return false;
//     }

//     if (interpreter_->AllocateTensors() != kTfLiteOk) {
//         Log::error("TfLiteEngine", "Failed to allocate tensors.");
//         return false;
//     }

//     // Store input tensor dimensions
//     TfLiteTensor* inputTensor = interpreter_->tensor(interpreter_->inputs()[0]);
//     if (inputTensor->dims->size >= 4) {
//         inputHeight_   = inputTensor->dims->data[1]; // H
//         inputWidth_    = inputTensor->dims->data[2]; // W
//         inputChannels_ = inputTensor->dims->data[3]; // C
//     } else {
//         inputHeight_ = inputWidth_ = inputChannels_ = 0;
//     }

//     Log::info("TfLiteEngine", "Model loaded and interpreter initialized.");
//     return true;
// }

// bool TfLiteEngine::resizeInputTensor(int width, int height, int channels) {
//     if (!interpreter_) return false;
//     if (interpreter_->ResizeInputTensor(0, {1, height, width, channels}) != kTfLiteOk) {
//         Log::error("TfLiteEngine", "Failed to resize input tensor.");
//         return false;
//     }
//     if (interpreter_->AllocateTensors() != kTfLiteOk) {
//         Log::error("TfLiteEngine", "Failed to re-allocate tensors after resizing.");
//         return false;
//     }

//     inputWidth_ = width;
//     inputHeight_ = height;
//     inputChannels_ = channels;
//     return true;
// }

// bool TfLiteEngine::setInputTensor(const std::vector<float>& inputData) {
//     if (!interpreter_) return false;
//     float* input = interpreter_->typed_input_tensor<float>(0);
//     if (!input) {
//         Log::error("TfLiteEngine", "Input tensor not available.");
//         return false;
//     }

//     std::copy(inputData.begin(), inputData.end(), input);
//     return true;
// }

// bool TfLiteEngine::invoke() {
//     if (!interpreter_) return false;
//     if (interpreter_->Invoke() != kTfLiteOk) {
//         Log::error("TfLiteEngine", "Failed to invoke interpreter.");
//         return false;
//     }
//     return true;
// }

// std::vector<float> TfLiteEngine::getOutputTensor(int index) const {
//     std::vector<float> output;
//     if (!interpreter_) return output;

//     float* data = interpreter_->typed_output_tensor<float>(index);
//     if (!data) return output;

//     TfLiteTensor* tensor = interpreter_->tensor(interpreter_->outputs()[index]);
//     int size = tensor->bytes / sizeof(float);

//     output.assign(data, data + size);
//     return output;
// }

// int TfLiteEngine::getInputTensorSize() const {
//     if (!interpreter_) return 0;
//     TfLiteTensor* tensor = interpreter_->tensor(interpreter_->inputs()[0]);
//     return tensor->bytes / sizeof(float);
// }

// int TfLiteEngine::getOutputTensorSize(int index) const {
//     if (!interpreter_ || index >= interpreter_->outputs().size()) return 0;
//     TfLiteTensor* tensor = interpreter_->output_tensor(index);
//     return tensor->bytes / sizeof(float);
// }

// // Add these methods to your existing TfLiteEngine.cpp file:

// int TfLiteEngine::getNumOutputs() const {
//     if (!interpreter_) {
//         return 0;
//     }
//     return interpreter_->outputs().size();
// }

// std::vector<int> TfLiteEngine::getOutputTensorShape(int index) const {
//     std::vector<int> shape;
    
//     if (!interpreter_ || index < 0 || index >= interpreter_->outputs().size()) {
//         return shape;
//     }
    
//     TfLiteTensor* tensor = interpreter_->output_tensor(index);
//     if (!tensor || !tensor->dims) {
//         return shape;
//     }
    
//     for (int i = 0; i < tensor->dims->size; i++) {
//         shape.push_back(tensor->dims->data[i]);
//     }
    
//     return shape;
// }

// } // namespace tflite
























// File: NeptuneFacialSDK/core/src/tflite/TfLiteEngine.cpp

// Robust TensorFlow Lite runtime wrapper implementation.
// Loads .tflite models, prepares interpreter, manages input/output tensors,
// and runs inference with clear error logging.


// File: NeptuneFacialSDK/core/src/tflite/TfLiteEngine.cpp
#include "neptune/TfLiteEngine.h"
#include "neptune/Log.h"

#include <cstring>
#include <algorithm>

namespace neptune {

TfLiteEngine::TfLiteEngine() = default;
TfLiteEngine::~TfLiteEngine() = default;
// Add this function implementation to your file
bool TfLiteEngine::isLoaded() const {
    // The engine is considered loaded and ready if both the model and interpreter are valid
    return model_ != nullptr && interpreter_ != nullptr;
}

bool TfLiteEngine::loadModel(const std::string& modelPath) {
    model_ = ::tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model_) {
        lastError_ = "Failed to load TFLite model from: " + modelPath;
        //NEP_LOGE("[TfLiteEngine] %s", lastError_.c_str());
        return false;
    }

    ::tflite::ops::builtin::BuiltinOpResolver resolver;
    ::tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    if (!interpreter_) {
        lastError_ = "Failed to create TFLite interpreter";
        return false;
    }

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        lastError_ = "AllocateTensors() failed";
        return false;
    }

    updateInputDims();
    //NEP_LOGI("[TfLiteEngine] Model loaded. Input HWC = %d x %d x %d",
          //   inputHeight_, inputWidth_, inputChannels_);
    return true;
}


bool TfLiteEngine::resizeInputTensor(int width, int height, int channels) {
    if (!interpreter_) {
        lastError_ = "Interpreter not initialized";
        return false;
    }
    const std::vector<int> dims = {1, height, width, channels};
    if (interpreter_->ResizeInputTensor(interpreter_->inputs()[0], dims) != kTfLiteOk) {
        lastError_ = "ResizeInputTensor() failed";
       // NEP_LOGE("[TfLiteEngine] %s", lastError_.c_str());
        return false;
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        lastError_ = "Re-AllocateTensors() after resize failed";
        //NEP_LOGE("[TfLiteEngine] %s", lastError_.c_str());
        return false;
    }
    updateInputDims();
    return true;
}

bool TfLiteEngine::setInputTensor(const std::vector<float>& inputData) {
    if (!interpreter_) {
        lastError_ = "Interpreter not initialized";
        return false;
    }
    if (interpreter_->inputs().empty()) {
        lastError_ = "No input tensors";
        return false;
    }

    TfLiteTensor* tensor = interpreter_->tensor(interpreter_->inputs()[0]);
    if (!tensor) {
        lastError_ = "Null input tensor";
        return false;
    }
    if (tensor->type != kTfLiteFloat32) {
        lastError_ = "Input tensor type is not float32";
        return false;
    }

    // Compute expected element count
    int64_t elems = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        elems *= tensor->dims->data[i];
    }

    if (static_cast<int64_t>(inputData.size()) != elems) {
        lastError_ = "Input size mismatch: got " + std::to_string(inputData.size())
                   + ", expected " + std::to_string(elems);
       // NEP_LOGE("[TfLiteEngine] %s", lastError_.c_str());
        return false;
    }

    std::memcpy(tensor->data.f, inputData.data(), elems * sizeof(float));
    return true;
}

bool TfLiteEngine::invoke() {
    if (!interpreter_) {
        lastError_ = "Interpreter not initialized";
        return false;
    }
    if (interpreter_->Invoke() != kTfLiteOk) {
        lastError_ = "Interpreter Invoke() failed";
        return false;
    }
    return true;
}

std::vector<float> TfLiteEngine::getOutputTensor(int index) const {
    std::vector<float> out;
    if (!interpreter_) return out;
    const auto& outs = interpreter_->outputs();
    if (index < 0 || index >= static_cast<int>(outs.size())) return out;

    const TfLiteTensor* t = interpreter_->tensor(outs[index]);
    if (!t || t->type != kTfLiteFloat32) return out;

    int64_t elems = 1;
    for (int i = 0; i < t->dims->size; ++i) elems *= t->dims->data[i];

    const float* data = t->data.f;
    if (!data || elems <= 0) return out;

    out.assign(data, data + elems);
    return out;
}

int TfLiteEngine::getInputTensorSize() const {
    if (!interpreter_ || interpreter_->inputs().empty()) return 0;
    const TfLiteTensor* t = interpreter_->tensor(interpreter_->inputs()[0]);
    if (!t) return 0;
    return t->bytes / static_cast<int>(sizeof(float));
}

int TfLiteEngine::getOutputTensorSize(int index) const {
    if (!interpreter_) return 0;
    const auto& outs = interpreter_->outputs();
    if (index < 0 || index >= static_cast<int>(outs.size())) return 0;
    const TfLiteTensor* t = interpreter_->tensor(outs[index]);
    if (!t) return 0;
    return t->bytes / static_cast<int>(sizeof(float));
}

int TfLiteEngine::getNumOutputs() const {
    if (!interpreter_) return 0;
    return static_cast<int>(interpreter_->outputs().size());
}

std::vector<int> TfLiteEngine::getOutputTensorShape(int index) const {
    std::vector<int> shape;
    if (!interpreter_) return shape;
    const auto& outs = interpreter_->outputs();
    if (index < 0 || index >= static_cast<int>(outs.size())) return shape;
    const TfLiteTensor* t = interpreter_->tensor(outs[index]);
    if (!t || !t->dims) return shape;
    shape.reserve(t->dims->size);
    for (int i = 0; i < t->dims->size; ++i) shape.push_back(t->dims->data[i]);
    return shape;
}

void TfLiteEngine::updateInputDims() {
    inputWidth_ = inputHeight_ = inputChannels_ = 0;
    if (!interpreter_ || interpreter_->inputs().empty()) return;
    const TfLiteTensor* t = interpreter_->tensor(interpreter_->inputs()[0]);
    if (!t || !t->dims || t->dims->size < 4) return;
    // Expect NHWC
    inputHeight_   = t->dims->data[1];
    inputWidth_    = t->dims->data[2];
    inputChannels_ = t->dims->data[3];
}

} // namespace neptune
