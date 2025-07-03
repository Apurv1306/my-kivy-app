[app]
# The title of your application
title = CamApp

# The unique package name for your application
package.name = camapp

# The domain for your package (e.g., org.kivy.yourpackage)
package.domain = org.cam.app

# The directory where your main.py and other app files are located
source.dir = .

# File extensions to include in the APK
source.include_exts = py,png,jpg,kv,atlas

# The version of your application
version = 0.1

# Application requirements
# - python3==3.10.10: Specific Python version.
# - kivy==2.2.1: The Kivy framework.
# - numpy: A dependency often used with image processing.
# - requests: For making HTTP requests.
# - opencv: Uncommented to include OpenCV.
#   WARNING: Including OpenCV can be challenging due to its native dependencies.
#   It might require specific Python-for-Android recipes or pre-built wheels.
#   If your app crashes after including OpenCV, this is likely the cause.
#   You may need to research specific P4A recipes or community solutions for OpenCV.
requirements = python3==3.10.10,kivy==2.2.1,numpy,requests,opencv

# Orientation of the application (portrait, landscape, sensor)
orientation = landscape

# Python version for macOS builds (not directly related to Android crashes)
osx.python_version = 3

# Fullscreen mode (1 for true, 0 for false)
fullscreen = 1

# Android permissions required by your application.
# IMPORTANT: 'android.no_permissions_required = 1' was removed as it
# contradicted the explicit permissions listed below, leading to runtime
# permission errors and crashes.
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

[buildozer]
# Log level for buildozer (0-5, 5 is most verbose, useful for debugging)
log_level = 5

# Warn if running as root (0 to disable, 1 to enable)
warn_on_root = 0

[android]
# Target Android API level (e.g., 31 for Android 12).
# This should be a recent, supported API level.
android.api = 31

# Target Android NDK API level.
# It's recommended to keep this aligned with or slightly lower than android.api
# for better compatibility, especially with native libraries.
# Changed from 21 to 26 for better compatibility with API 31.
android.ndk_api = 26

# Architectures to build for. arm64-v8a is for 64-bit devices,
# armeabi-v7a for 32-bit devices.
android.archs = arm64-v8a,armeabi-v7a

# If you need to include specific Java files or libraries, you can specify them here.
# android.add_libs_armeabi-v7a =
# android.add_libs_arm64-v8a =
# android.add_src =
