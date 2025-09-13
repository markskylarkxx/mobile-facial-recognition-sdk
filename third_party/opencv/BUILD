cc_library(
    name = "opencv",
    hdrs = glob(["include/**/*.hpp", "include/**/*.h"]),
    includes = ["include"],
    linkopts = glob(["lib/*.dylib"]),  # Links the dynamic libraries
    visibility = ["//visibility:public"],
)
