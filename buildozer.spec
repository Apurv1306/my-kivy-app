[buildozer]
# Log level: 0 (error), 1 (info), 2 (debug)
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
source.include_exts = py,png,jpg,mp3,json,xml

# Python and Kivy requirements, plus OpenCV and requests
requirements = python3,kivy==2.0.0,opencv,requests

# Application category
category = other

# Android permissions
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# Target Android API level (e.g., 33 or 34 for recent Google Play requirements)
android.api = 33

# Minimum Android API level
android.minapi = 21

# Android architectures to build for
android.archs = arm64-v8a, armeabi-v7a

# Application orientation
orientation = landscape 

# Fullscreen mode (0 = no, 1 = yes)
fullscreen = 0 

# Presplash background color (RGBA hex code)
presplash.color = #000000FF

# Optional: Path to custom icon and presplash image (uncomment to use)
# icon.filename = %(source.dir)s/icon.png
# presplash.filename = %(source.dir)s/presplash.png

# Optional: Keystore details for signing (uncomment for Play Store publishing)
# android.keystore_alias = myappkey
# android.keystore_pass = your_keystore_password
# android.key_pass = your_key_password
