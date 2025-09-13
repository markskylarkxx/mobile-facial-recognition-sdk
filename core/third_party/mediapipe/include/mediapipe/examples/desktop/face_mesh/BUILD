# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

# CPU build for face mesh
cc_binary(
    name = "face_mesh_cpu",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
    ],
    copts = ["-I/opt/homebrew/opt/opencv/include/opencv4"],
    linkopts = [
        "-L/opt/homebrew/opt/opencv/lib",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_highgui",
        "-lopencv_videoio",   # required for VideoCapture/VideoWriter
        "-lopencv_video"      # sometimes needed for certain video functions
    ],
)

# BUILD for TFLite CPU
cc_binary(
    name = "face_mesh_tflite",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_calculators",
    ],
)

# Linux only GPU build
cc_binary(
    name = "face_mesh_gpu",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
    ],
)
