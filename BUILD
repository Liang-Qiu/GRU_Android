# Description:
#   TensorFlow LiangQiu word predictor app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", 
     "tf_copts",
     "tf_opts_nortti_if_android")

exports_files(["LICENSE"])

LINKER_SCRIPT = "//tensorflow/contrib/android:jni/version_script.lds"

cc_binary(
    name = "libtensorflow_demo.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]) + [],
    copts = tf_copts(),
    linkopts = [
        "-landroid",
        "-ljnigraphics",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--icf=all",  # Identical Code Folding
        "-Wl,--version-script",  # This line must be directly followed by LINKER_SCRIPT.
        LINKER_SCRIPT,
    ],
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":demo_proto_lib_cc",
        "//tensorflow/contrib/android:android_tensorflow_inference_jni",
        "//tensorflow/core:android_tensorflow_lib",
        LINKER_SCRIPT,
    ],
)

cc_library(
    name = "tensorflow_native_libs",
    srcs = [":libtensorflow_demo.so"],
    tags = [
        "manual",
        "notap",
    ],
)

android_binary(
    name = "word_predictor",
    srcs = glob([
        "src/**/*.java",
    ]),
    assets = glob(["assets/**"]),
    assets_dir = "assets",
    custom_package = "edu.ucla.liangqiu.predictor",
    inline_constants = 1,
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
        "//tensorflow/contrib/android:android_tensorflow_inference_java",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
            "gradleBuild/**",
            "libs/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "java_files",
    srcs = glob(["src/**/*.java"]),
)

filegroup(
    name = "jni_files",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
)

filegroup(
    name = "resource_files",
    srcs = glob(["res/**"]),
)

exports_files(["AndroidManifest.xml"])

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "demo_proto_lib",
    srcs = glob(
        ["**/*.proto"],
    ),
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
