#builds small test apps like your preprocess_test
#NeptuneFacialSDK/core/tests/CMakeLists.txt

# # Build preprocess test
# add_executable(preprocess_test test_preprocess.cpp)

# # Link OpenCV
# target_link_libraries(preprocess_test neptune_core ${OpenCV_LIBS})

# Preprocess test
# add_executable(preprocess_test test_preprocess.cpp)
# target_link_libraries(preprocess_test neptune_core ${OpenCV_LIBS})


# # Face detection test
add_executable(face_detection_test face_detection_test.cpp)
target_link_libraries(face_detection_test neptune_core ${OpenCV_LIBS})


# Add a new executable for real-time camera testing
add_executable(real_time_detection_test real_time_detection_test.cpp)

# Link the new executable with the NeptuneCore library and required OpenCV modules
target_link_libraries(real_time_detection_test neptune_core ${OpenCV_LIBS})
     # This should link all components found by find_package


     add_executable(main_webrtc  main_webrtc.cpp )
target_link_libraries(main_webrtc neptune_core ${OpenCV_LIBS})





# mkdir build && cd build
# cmake ..
# make  make -j4

# ./preprocess_test
