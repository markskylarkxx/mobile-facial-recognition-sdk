load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "mediapipe_face_detector",
    srcs = ["MediaPipeFaceDetector.cpp"],
    hdrs = ["//core/include/neptune:MediaPipeFaceDetector.h"],
    copts = ["-std=c++17", "-I/opt/homebrew/Cellar/opencv/4.12.0_10/include/opencv4"],
    linkopts = ["-L/opt/homebrew/Cellar/opencv/4.12.0_10/lib", "-lopencv_core", "-lopencv_imgproc", "-lopencv_highgui", "-lopencv_imgcodecs"],
    deps = [
        "@mediapipe//tasks/cc/vision/face_landmarker:face_landmarker_cc",
        "@mediapipe//mediapipe/framework:framework",
        "@mediapipe//mediapipe/framework/formats:image",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_format",
        "@mediapipe//mediapipe/framework/port",
        "@com_google_absl//absl:base",
        "@com_google_absl//absl:strings",
        "@com_google_protobuf//:protobuf",
    ],
)
