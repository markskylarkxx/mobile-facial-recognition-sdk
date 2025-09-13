// jni_bridge.cpp

#include <jni.h>
#include <opencv2/opencv.hpp>
#include "neptune/NeptuneSDK.h"

extern "C" {

// 1. Create SDK
JNIEXPORT jlong JNICALL
Java_com_neptune_sdk_FaceSDK_nativeCreate(
    JNIEnv* env,
    jobject thiz,
    jstring faceModel,
    jstring emotionModel) {

    const char* f = env->GetStringUTFChars(faceModel, nullptr);
    const char* e = env->GetStringUTFChars(emotionModel, nullptr);

    neptune::NeptuneConfig cfg;
    cfg.faceDetectionModelPath = f;
    cfg.emotionModelPath = e;

    auto sdk = neptune::NeptuneSDK::create(cfg);

    env->ReleaseStringUTFChars(faceModel, f);
    env->ReleaseStringUTFChars(emotionModel, e);

    return reinterpret_cast<jlong>(sdk.release());
}

// 2. Process image
JNIEXPORT jobjectArray JNICALL
Java_com_neptune_sdk_FaceSDK_nativeProcessImage(
    JNIEnv* env,
    jobject thiz,
    jlong handle,
    jbyteArray imageBytes,
    jint width,
    jint height) {

    auto* sdk = reinterpret_cast<neptune::NeptuneSDK*>(handle);
    if (!sdk) return nullptr;

    jbyte* bytes = env->GetByteArrayElements(imageBytes, nullptr);
    cv::Mat image(height, width, CV_8UC3, (unsigned char*)bytes);

    std::vector<neptune::NeptuneResult> results = sdk->processImage(image);

    env->ReleaseByteArrayElements(imageBytes, bytes, JNI_ABORT);

    // TODO: Convert results → Java objects (FaceDetectionResult, EmotionResult, etc.)
    // For now just return null or a placeholder.
    return nullptr;
}

// 3. Release
JNIEXPORT void JNICALL
Java_com_neptune_sdk_FaceSDK_nativeRelease(
    JNIEnv* env,
    jobject thiz,
    jlong handle) {
    auto* sdk = reinterpret_cast<neptune::NeptuneSDK*>(handle);
    delete sdk;
}

} // extern "C"
