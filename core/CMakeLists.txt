# cmake_minimum_required(VERSION 3.10)
# project(NeptuneFacialSDK)

# set(CMAKE_CXX_STANDARD 17)

# # Find OpenCV
# #find_package(OpenCV REQUIRED)
# find_package(OpenCV REQUIRED COMPONENTS core videoio highgui)

# # Get the parent directory (where third_party is located)
# get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

# # Collect all core sources
# file(GLOB_RECURSE CORE_SOURCES src/*.cpp)

# # Build the core Neptune library
# add_library(neptune_core ${CORE_SOURCES})

# # Include paths - Use the parent directory for third_party
# target_include_directories(neptune_core PUBLIC
#   ${CMAKE_SOURCE_DIR}/include
#   ${OpenCV_INCLUDE_DIRS}
#   ${PARENT_DIR}/third_party  # ← This is the correct path now
#   ${PARENT_DIR}/third_party/tensorflow
#   ${PARENT_DIR}/third_party/mediapipe
#   ${PARENT_DIR}/third_party/third_party/absl



# )

# # Debug: Show the actual paths
# message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
# message(STATUS "PARENT_DIR: ${PARENT_DIR}")
# message(STATUS "Third party path: ${PARENT_DIR}/third_party")

# # Link OpenCV to the library
# target_link_libraries(neptune_core ${OpenCV_LIBS})

# # Link TensorFlow Lite dylib
# target_link_libraries(neptune_core
#   ${CMAKE_SOURCE_DIR}/lib/libtensorflowlite.dylib
# )

# # Add tests subdirectory
# add_subdirectory(tests)




#..................................................................................
cmake_minimum_required(VERSION 3.10)
project(NeptuneFacialSDK)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core videoio highgui imgproc)

# Get the parent directory (where third_party is located)
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

# Collect all core sources
file(GLOB_RECURSE CORE_SOURCES 
    src/*.cpp
    src/MediaPipeLandmarks.cpp  # Add this
)

# Build the core Neptune library
add_library(neptune_core ${CORE_SOURCES})

# MediaPipe option
option(NEPTUNE_USE_MEDIAPIPE "Build with MediaPipe support" OFF)

# Include paths
target_include_directories(neptune_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PARENT_DIR}/third_party
    ${PARENT_DIR}/third_party/tensorflow
    ${PARENT_DIR}/third_party/mediapipe
)

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

# Link OpenCV to the library
target_link_libraries(neptune_core ${OpenCV_LIBS})

# Link TensorFlow Lite dylib
target_link_libraries(neptune_core
    ${CMAKE_SOURCE_DIR}/lib/libtensorflowlite.dylib
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
