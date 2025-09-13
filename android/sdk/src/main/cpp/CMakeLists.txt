cmake_minimum_required(VERSION 3.10.2)
project(neptune_sdk)

# Add the core library from the root directory
add_subdirectory(../../../../core core)

# Find OpenCV (assuming it's in third_party)
find_package(OpenCV REQUIRED PATHS ../../../../third_party/opencv)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find JNI
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Create the JNI library
add_library(neptune_sdk SHARED
    jni_bridge.cpp
)

target_include_directories(neptune_sdk PRIVATE
    ${JNI_INCLUDE_DIRS}
    ../../../../core/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(neptune_sdk
    neptune_core
    ${JNI_LIBRARIES}
    ${OpenCV_LIBS}
    log
    # Link against prebuilt TFLite
)

# Copy models from core/models to assets during build