

#include "neptune/WebRTCManager.h"
#include <iostream>
// main_webrtc.cpp
int main() {
    std::cout << "Starting Neptune WebRTC Server..." << std::endl;
    WebRTCManager manager;

    // TEMPORARY: Test with webcam instead of WebRTC
    manager.testWithLocalWebcam();

    return 0;
}