workspace(name = "facial_recognition_sdk")

# Optional: rules_cc for Bazel C++ support
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cc",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.0.5.zip"],
    strip_prefix = "rules_cc-0.0.5",
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")
rules_cc_dependencies()
