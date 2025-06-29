[buildozer]
# (int) Log level (0 = error only, 1 = info, 2 = debug (full log))
log_level = 2

# (int) If set to 1, the app will be built in debug mode.
# debug = 0 # Uncomment and set to 1 if you want to explicitly enable debug mode here.

# (int) If set to 1, will warn you if you build as root
warn_on_root = 1

[app]

# (str) Title of your application
title = FaceApp Attendance

# (str) Package name
package.name = faceappattendance

# (str) Package domain (needed for Android/iOS packaging)
package.domain = org.yourcompany  # IMPORTANT: Change 'yourcompany' to a unique identifier for your app

# (str) Application versioning (method 1)
version = 0.1

# (str) Source code where the main.py lives
source.dir = .

# (list) List of all the files to include in your APK package, separated by commas.
# This is crucial for all your assets and the Haar cascade XML file.
source.include_exts = py,png,jpg,mp3,json,xml

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy==2.0.0,opencv,requests

# (str) The category of the application:
# Available categories are:
#   'audio', 'video', 'office', 'game', 'other'
category = other

# (list) Permissions
# (https://developer.android.com/reference/android/Manifest.permission.html)
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# (int) Target Android API level. For Google Play Store, this needs to be high.
# As of current guidelines, API level 33 or 34 is generally required for new apps.
android.api = 33

# (int) Minimum Android API level supported by your application
android.minapi = 21

# (list) Android device archs to build for, comma-separated.
# 'arm64-v8a' and 'armeabi-v7a' cover most modern Android devices.
android.archs = arm64-v8a, armeabi-v7a

# (int) Set the orientation of the application (e.g. landscape, portrait)
# Set to landscape as requested
orientation = landscape 

# (int) Whether your application should be fullscreen or not
fullscreen = 0 # Set to 0 for debugging, can be 1 for production

# (str) Presplash background color in rgba format (e.g. #FFFFFFFF)
# This will be the background color of the first screen while your app loads.
presplash.color = #000000FF

# (str) Path to a custom icon. If not specified, default Kivy icon will be used.
# icon.filename = %(source.dir)s/icon.png

# (str) Path to a custom presplash image. If not specified, default Kivy presplash will be used.
# presplash.filename = %(source.dir)s/presplash.png

# (str) Gmail username and password for Play Store Publishing
# If not set, you will need to manually sign your APK.
# android.keystore_alias = myappkey
# android.keystore_pass = your_keystore_password
# android.key_pass = your_key_password

# The default values for the next options are usually fine, but you can uncomment and adjust if needed:
# android.ndk = 25b
# android.sdk = 2022.10.10
# android.java_home = /usr/lib/jvm/java-17-openjdk-amd64
