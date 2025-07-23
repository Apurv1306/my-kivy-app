[app]
# ---------------------------------------------------------
# Basic app metadata
# ---------------------------------------------------------
title = FaceApp
package.name = faceapp
package.domain = org.example               # change to your domain
version = 0.1
# If you keep the default icon, Buildozer generates one automatically.
# icon.filename = data/icon.png

# ---------------------------------------------------------
# Source files to include in the APK
# ---------------------------------------------------------
source.dir = .
source.include_exts = py,kv,png,jpg,jpeg,wav,txt,json,xml
# Exclude read-me files, tests, etc. to keep the APK small
source.exclude_dirs = tests, .git, __pycache__

# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
entrypoint = flaskapk_offline_safe.py      # rename if you change the filename
orientation = portrait
fullscreen = 0

# ---------------------------------------------------------
# Python for Android build settings
# ---------------------------------------------------------
# Tested with Python-for-Android master and NDK r26c
requirements = \
    python3==3.11, \
    kivy==2.2.1, \
    plyer, \
    flask, \
    flask_cors, \
    requests, \
    numpy, \
    opencv, \
    certifi

# Pin the current master branch of python-for-android (most reliable for Kivy 2.2+)
p4a.branch = master

# OpenCV needs extra build flags for SIMD/NEON on ARM
android.ndk = 26b
android.api = 34
ndk_api = 21                     # 21 is the minimum that ships with libc++
android.archs = armeabi-v7a, arm64-v8a

# ---------------------------------------------------------
# Permissions & hardware features
# ---------------------------------------------------------
android.permissions = \
    CAMERA, \
    INTERNET, \
    WRITE_EXTERNAL_STORAGE, \
    READ_EXTERNAL_STORAGE, \
    WAKE_LOCK, \
    RECORD_AUDIO

# For devices that lack a camera, the app will be hidden in Google Play
android.features = android.hardware.camera.autofocus, android.hardware.camera

# ---------------------------------------------------------
# Java/Kotlin & Gradle tweaks
# ---------------------------------------------------------
# Use the modern Android Gradle Plugin
android.gradle_dependencies = \
    com.google.android.material:material:1.11.0

# ---------------------------------------------------------
# Optimisations & size trimming
# ---------------------------------------------------------
android.disable_legacy_sdktools = 1
# Strip debug symbols to reduce APK size
android.strip_mode = all
# Remove empty locale files that pip brings along
android.extra_dist_dirs = assets

# ---------------------------------------------------------
# Release keystore (auto-signed debug APK if left blank)
# ---------------------------------------------------------
# keys.properties path is recommended instead of hard-coding sensitive data.
# keystore = ~/.keystores/faceapp-release.jks
# keyalias = faceapp
# keystorepw = ***
# keyaliaspw = ***

# ---------------------------------------------------------
# Logging & debug helpers
# ---------------------------------------------------------
log_level = 2
# Uncomment while developing; comment for release builds to improve performance
# android.logcat_args = *:I
