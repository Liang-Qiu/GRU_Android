# TensorFlow Android Word Prediction Demo

This folder contains a simple word prediction demo application utilizing TensorFlow.

## Description

This demo uses a 2-layer GRU model to predict words according to the text history cognitinized by
the Google voice recognition service in real-time, displaying the top results in the main acticity.

## To build/install/run

As a prerequisite, Bazel, the Android NDK, and the Android SDK must all be
installed on your system.

1. Get the recommended Bazel version listed at:
        https://www.tensorflow.org/versions/master/get_started/os_setup.html#source
2. The Android NDK may be obtained from:
        http://developer.android.com/tools/sdk/ndk/index.html
3. The Android SDK and build tools may be obtained from:
        https://developer.android.com/tools/revisions/build-tools.html

Copy this folder and paste it under the route tensorflow/tensorflow/examples/. 

The Android entries in [`<workspace_root>/WORKSPACE`](../../../WORKSPACE#L2-L13) must be
uncommented with the paths filled in appropriately depending on where you
installed the NDK and SDK. Otherwise an error such as:
"The external label '//external:android/sdk' is not bound to anything" will
be reported.

The TensorFlow `GraphDef` that contains the model definition and weights
is not packaged in the repo because of its size. Instead, you must
first download the file to the `assets` directory in the source tree:

The vocab file describing the possible prediction will also be in the
assets directory.

Then, after editing your WORKSPACE file, you must build the APK. Run this from
your workspace root:

```bash
$ bazel build //tensorflow/examples/android:liangqiu_predictor
```

If you get build errors about protocol buffers, run
`git submodule update --init` and build again.

If adb debugging is enabled on your Android 5.0 or later device, you may then
use the following command from your workspace root to install the APK once
built:

```bash
$ adb install -r -g bazel-bin/tensorflow/examples/android/liangqiu_predictor.apk
```

Some older versions of adb might complain about the -g option (returning:
"Error: Unknown option: -g").  In this case, if your device runs Android 6.0 or
later, then make sure you update to the latest adb version before trying the
install command again. If your device runs earlier versions of Android, however,
you can issue the install command without the -g option.

Alternatively, a streamlined means of building, installing and running in one
command is:

```bash
$ bazel mobile-install //tensorflow/examples/android:liangqiu_predicor --start_app
```

If voice record permission errors are encountered (possible on Android Marshmallow or
above), then the `adb install` command above should be used instead, as it
automatically grants the required voice record permissions with `-g`. The permission
errors may not be obvious if the app halts immediately, so if you installed
with bazel and the app doesn't come up, then the easiest thing to do is try
installing with adb.

Once the app is installed it will be named "Word Predictor" and have the orange
TensorFlow logo as its icon.
