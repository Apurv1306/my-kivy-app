[buildozer]
# Log level: 0 (error), 1 (info), 2 (debug - highly recommended for debugging)
log_level = 2
warn_on_root = 1

[app]
# Application title
title = FaceApp Attendance

# Package identifier (e.g., com.yourcompany.yourapp)
package.name = faceappattendance
package.domain = org.yourcompany  # IMPORTANT: Change 'yourcompany' to your unique domain

# Application version
version = 0.1

# Source code directory (usually the current directory)
source.dir = .

# Files to include in the APK (Python files, assets, etc.)
# Ensure all your Python files, the .xml cascade, and media files are included.
source.include_exts = py,png,jpg,mp3,json,xml

# Python and Kivy requirements, plus OpenCV and requests
# Keep Kivy version fixed for consistency.
# opencv is essential for computer vision.
requirements = python3,kivy==2.0.0,opencv,requests

# The category of the application
category = other

# Android permissions - essential for camera, internet, and internal storage access.
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# Target Android API level. For Google Play Store, this needs to be high.
# API level 33 or 34 are current common requirements.
android.api = 33

# Minimum Android API level supported by your application
android.minapi = 21

# Android device architectures to build for. arm64-v8a is dominant.
android.archs = arm64-v8a, armeabi-v7a

# Set the orientation of the application
orientation = landscape 

# Whether your application should be fullscreen or not (0=no, 1=yes)
fullscreen = 0 

# Presplash background color (RGBA hex code)
presplash.color = #000000FF

# Increase build timeout for potentially long compilation steps or slow downloads
# Default is 5 minutes, 10-15 minutes might be needed for fresh builds.
# This helps prevent GitHub Actions from prematurely cancelling the job.
app.build_timeout = 900 # 900 seconds = 15 minutes

# Add common Android Gradle dependencies if needed.
# This can sometimes resolve missing library errors during the Android part of the build.
# You might need to add specific versions depending on your `android.api`
android.gradle_dependencies = \
    'androidx.core:core-ktx:1.9.0',\
    'androidx.appcompat:appcompat:1.6.1'

# Enable multidex for larger applications to prevent DEX limit issues.
# OpenCV can make your app large enough to require this.
android.enable_multidex = 1

# Optional: You can specify exact NDK and SDK versions if `stable` gives issues
# android.ndk = 25b # Example: NDK version
# android.sdk = 2022.10.10 # Example: Android SDK build-tools version
# android.java_home = /usr/lib/jvm/java-17-openjdk-amd64 # Example: Java path

# Optional: Path to custom icon and presplash image (uncomment to use)
# icon.filename = %(source.dir)s/icon.png
# presplash.filename = %(source.dir)s/presplash.png

# Optional: Keystore details for signing (uncomment for Play Store publishing)
# android.keystore_alias = myappkey
# android.keystore_pass = your_keystore_password
# android.key_pass = your_key_password
