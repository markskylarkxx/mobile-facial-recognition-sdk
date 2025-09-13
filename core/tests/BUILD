load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "face_detection_test",
    srcs = ["face_detection_test.cpp"],
    deps = [
        "//core/src:mediapipe_face_detector",
    ],
    copts = ["-std=c++17", "-I/opt/homebrew/Cellar/opencv/4.12.0_10/include/opencv4"],
    linkopts = ["-L/opt/homebrew/Cellar/opencv/4.12.0_10/lib", "-lopencv_core", "-lopencv_imgproc", "-lopencv_highgui", "-lopencv_imgcodecs"],
)
