
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
   # ${PARENT_DIR}/third_party/mediapipe
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







#####################################################################################

# cmake_minimum_required(VERSION 3.10) # Use 3.12 to ensure GLOB_RECURSIVE support
# project(NeptuneFacialCore)
# set(CMAKE_CXX_STANDARD 17)

# # Collect all core sources
# file(GLOB_RECURSE CORE_SOURCES
#     src/*.cpp
#     src/MediaPipeLandmarks.cpp
# )

# # Build the core Neptune library
# add_library(neptune_core ${CORE_SOURCES})

# # Force MediaPipe ON by default
# set(NEPTUNE_USE_MEDIAPIPE ON CACHE BOOL "Build with MediaPipe support" FORCE)

# # Include paths
# target_include_directories(neptune_core PUBLIC
#     ${CMAKE_SOURCE_DIR}/../include # TensorFlow Lite headers (tensorflow/lite)
#     ${CMAKE_SOURCE_DIR}/include # Neptune headers (neptune/)
#     ${CMAKE_SOURCE_DIR}/../third_party/opencv/include # OpenCV headers
# )

# # MediaPipe toggle
# if(NEPTUNE_USE_MEDIAPIPE)
#     target_compile_definitions(neptune_core PUBLIC NEPTUNE_USE_MEDIAPIPE)
#     message(STATUS "Building with MediaPipe support")
# else()
#     message(STATUS "Building without MediaPipe support (simulation mode)")
# endif()

# # Debug messages
# message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# # Link TensorFlow Lite dylib
# find_library(TFLITE_LIBRARY
#     NAMES tensorflowlite libtensorflowlite
#     PATHS
#         ${CMAKE_SOURCE_DIR}/lib
#         /opt/homebrew/lib
#         /usr/local/lib
# )

# if(NOT TFLITE_LIBRARY)
#     message(FATAL_ERROR "TensorFlow Lite library not found at ${CMAKE_SOURCE_DIR}/lib")
# endif()

# target_link_libraries(neptune_core PRIVATE ${TFLITE_LIBRARY})

# # Link vendored OpenCV libraries
# find_library(OPENCV_CORE NAMES opencv_core libopencv_core PATHS ${CMAKE_SOURCE_DIR}/../third_party/opencv/lib)
# find_library(OPENCV_IMGPROC NAMES opencv_imgproc libopencv_imgproc PATHS ${CMAKE_SOURCE_DIR}/../third_party/opencv/lib)
# find_library(OPENCV_HIGHGUI NAMES opencv_highgui libopencv_highgui PATHS ${CMAKE_SOURCE_DIR}/../third_party/opencv/lib)
# find_library(OPENCV_VIDEOIO NAMES opencv_videoio libopencv_videoio PATHS ${CMAKE_SOURCE_DIR}/../third_party/opencv/lib)
# find_library(OPENCV_IMGCODECS NAMES opencv_imgcodecs libopencv_imgcodecs PATHS ${CMAKE_SOURCE_DIR}/../third_party/opencv/lib)

# if(NOT OPENCV_CORE OR NOT OPENCV_IMGPROC OR NOT OPENCV_HIGHGUI OR NOT OPENCV_VIDEOIO OR NOT OPENCV_IMGCODECS)
#     message(FATAL_ERROR "One or more OpenCV libraries not found in ${CMAKE_SOURCE_DIR}/../third_party/opencv/lib")
# endif()

# target_link_libraries(neptune_core PRIVATE
#     ${OPENCV_CORE}
#     ${OPENCV_IMGPROC}
#     ${OPENCV_HIGHGUI}
#     ${OPENCV_VIDEOIO}
#     ${OPENCV_IMGCODECS}
# )

# # Add tests subdirectory 
# add_subdirectory(tests)































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
