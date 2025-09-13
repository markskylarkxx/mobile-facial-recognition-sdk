<!--
This document provides a comprehensive overview of the Neptune Facial SDK project structure.
It is intended to be a single source of truth for the purpose and content of each folder and file.
-->

NeptuneFacialSDK/
<!-- The root directory for the entire project. -->

This is the main directory that contains all components of the cross-platform SDK, including the core C++ engine, platform-specific wrappers, models, and build tools.

core/

<!-- This folder contains the platform-agnostic C++ engine. -->

The core/ folder is the heart of the SDK. All the core business logic, such as facial detection and emotion recognition, is implemented here in C++. This is done so the code can be reused across all supported platforms (Android, iOS, etc.), making it easier to maintain and ensuring consistent performance.

core/include/neptune/

This sub-folder contains the public header files that define the SDK's C++ API. Any other part of the SDK that needs to use the core logic (like the platform wrappers) will include these files.

NeptuneSDK.h: This is the main public façade for the SDK. It defines the high-level functions that developers will call to perform a complete analysis (e.g., detect a face, recognize emotion, and check for liveness in a single call).

FaceDetector.h: Defines the interface for the facial detection component.

EmotionRecognizer.h: Defines the interface for the emotion recognition component.

LivenessChecker.h: Defines the interface for the liveness check component.

Types.h: Defines all the fundamental data structures and enumerations (like FaceBox, Emotion, and NeptuneResult) that are used throughout the entire SDK.

core/src/

This sub-folder contains the private C++ implementation files for the core logic. These files contain the actual code that makes the SDK work. They are not exposed to the public API.

NeptuneSDK.cpp: The implementation for the main SDK class.

FaceDetector.cpp: The implementation for the face detection logic.

EmotionRecognizer.cpp: The implementation for the emotion recognition logic.

LivenessChecker.cpp: The implementation for the liveness check logic.

tflite/TfLiteEngine.cpp: Contains the code for loading and running the TensorFlow Lite models. This is where the core machine learning inference happens.

img/Preprocess.cpp: Contains utility functions for preparing image data for the models (e.g., resizing, normalization).

util/Log.cpp: Contains logging functions for debugging and reporting internal status.

third_party/

<!-- This folder holds prebuilt external libraries that the SDK depends on. -->

This folder is for external dependencies that you don't build yourself. For a C++ project, this is a standard practice to manage precompiled libraries like TensorFlow Lite and OpenCV that are specific to a platform's architecture.

android/: Contains the prebuilt .so files for Android.

ios/: Contains the prebuilt .xcframework files for iOS.

models/

<!-- This folder is the single source of truth for the machine learning models. -->

The models/ directory stores all the .tflite model files and label files that the SDK uses. Keeping them in a single, top-level location prevents versioning issues and ensures all platforms use the same models.

android/

<!-- This folder contains the code for the Android AAR library. -->

This folder holds all the platform-specific code and resources for the Android SDK. It includes the wrapper that exposes the C++ core to Kotlin/Java, as well as a sample application.

sdk/: The code that produces the .AAR file, which is the library format Android developers will use.

demo-app/: A fully functional Android application that demonstrates how to integrate and use the SDK.

ios/

<!-- This folder contains the code for the iOS XCFramework. -->

This folder holds all the platform-specific code and resources for the iOS SDK. It includes the wrapper that exposes the C++ core to Swift/Objective-C, as well as a sample application.

NeptuneFaceSDK/: The code that produces the .xcframework file, which is the library format iOS developers will use.

demo-app/: A fully functional iOS application that demonstrates how to integrate and use the SDK.

tools/

<!-- This folder contains all the helper scripts for development and building. -->

The tools/ directory is for automation. The scripts here handle repetitive tasks like copying models to their correct locations and running build commands for the different platforms.

docs/

<!-- This folder contains all the user-facing documentation. -->

This folder is dedicated to user documentation, such as integration guides and a public API reference, that will help developers use your SDK.

CMakeLists.txt

<!-- This is the main build configuration file for the C++ parts of the project. -->

This is a critical file that tells CMake how to build the entire project, including the core library and any platform-specific C++ components.

