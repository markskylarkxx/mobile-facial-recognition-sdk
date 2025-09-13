#Root CMakeLists.txt just orchestrates everything.

cmake_minimum_required(VERSION 3.14)
project(NeptuneFacialSDK)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_subdirectory(core)
add_subdirectory(tests)
