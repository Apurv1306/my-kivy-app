[app]
# (str) Title of your application
title = FaceApp

# (str) Package name
package.name = faceapp

# (str) Package domain (needed for android packaging)
package.domain = com.example

# (str) Source code where the main.py lives
source.dir = .

# (list) Source files to include (extensions)
source.include_exts = py,png,jpg,mp3,xml

# (list) List of inclusions using pattern matching
source.include_patterns = *.png,*.jpg,*.mp3,*.xml

# (str) Application versioning (method 1)
version = 1.0

# (list) Application requirements
# Include python3, Kivy, OpenCV, NumPy, Requests, OpenSSL for SMTP TLS
requirements = python3,kivy==2.3.0,opencv-python==4.12.0,numpy==2.3.2,requests==2.31.0

# (str) Presplash of the application
# presplash.filename = %(source.dir)s/presplash.png

# (str) Icon of the application
# icon.filename = %(source.dir)s/icon.png

# (str) Supported orientation (one of landscape, portrait or all)
orientation = portrait

# (list) Permissions
android.permissions = CAMERA,INTERNET

# (int) Android API to use
android.api = 31

# (int) Minimum API required
android.minapi = 21

# (str) Android entry point, default is fine for Kivy apps
android.entrypoint = org.kivy.android.PythonActivity

# (list) List of Java .jar files to add to libs for pyjnius (if any)
# android.add_jars =

# (list) Android application meta-data to set (key=value)
# android.meta_data =

[buildozer]
# (int) Log level (0=error, 1=warning, 2=info, 3=debug)
log_level = 2

# (bool) Warn when running as root
warn_on_root = 1

# (str) Path for build artifacts
# build_dir = ./.buildozer

# (str) Path for final APK
# bin_dir = ./bin
