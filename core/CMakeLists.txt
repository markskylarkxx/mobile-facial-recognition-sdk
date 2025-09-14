
cmake_minimum_required(VERSION 3.10)
project(NeptuneFacialSDK)

set(CMAKE_CXX_STANDARD 17)

# Get the parent directory (where third_party is located)
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

# Collect all core sources
file(GLOB_RECURSE CORE_SOURCES 
    src/*.cpp
    src/MediaPipeLandmarks.cpp  
)

# Build the core Neptune library
add_library(neptune_core ${CORE_SOURCES})

# Force MediaPipe ON by default (no need to pass -D flag)
set(NEPTUNE_USE_MEDIAPIPE ON CACHE BOOL "Build with MediaPipe support" FORCE)

# Include paths
target_include_directories(neptune_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${PARENT_DIR}/third_party/opencv/include   # OpenCV headers
    ${PARENT_DIR}/third_party
    ${PARENT_DIR}/third_party/tensorflow
    ${PARENT_DIR}/third_party/mediapipe
)

# MediaPipe toggle
if(NEPTUNE_USE_MEDIAPIPE)
    target_compile_definitions(neptune_core PUBLIC NEPTUNE_USE_MEDIAPIPE)
    message(STATUS "Building with MediaPipe support")
else()
    message(STATUS "Building without MediaPipe support (simulation mode)")
endif()

# Debug: Show the actual paths
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "PARENT_DIR: ${PARENT_DIR}")
message(STATUS "Third party path: ${PARENT_DIR}/third_party")

# Link TensorFlow Lite dylib
target_link_libraries(neptune_core
    ${CMAKE_SOURCE_DIR}/lib/libtensorflowlite.dylib
)

# Link vendored OpenCV libraries
target_link_libraries(neptune_core
    ${PARENT_DIR}/third_party/opencv/lib/libopencv_core.dylib
    ${PARENT_DIR}/third_party/opencv/lib/libopencv_imgproc.dylib
    ${PARENT_DIR}/third_party/opencv/lib/libopencv_highgui.dylib
    ${PARENT_DIR}/third_party/opencv/lib/libopencv_videoio.dylib
    ${PARENT_DIR}/third_party/opencv/lib/libopencv_imgcodecs.dylib
)

# Add tests subdirectory
add_subdirectory(tests)













#libneptune_core.a (your static library for the SDK) was built.
#./tests/face_detection_test



# Key points:
# Make sure you have MediaPipe C++ Tasks API installed and linked in your build.
# The FaceLandmarker model should exist (face_landmarker_task_cpu.tflite) or provide your custom path.
# This implementation now populates FaceBox.landmarks, which your liveness checker needs.
# Debug logs in FaceDetector.cpp will show the number of landmarks detected.


#Then from your project root (where the WORKSPACE file is):
#bazel build //core/tests:face_detection_test
#bazel run //core/tests:face_detection_test -- --image path/to/test.jpg
