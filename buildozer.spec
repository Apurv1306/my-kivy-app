[app]
title = CamApp
package.name = camapp
package.domain = org.cam.app
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3==3.10.10,kivy==2.2.1,opencv,numpy,requests
orientation = portrait
osx.python_version = 3
fullscreen = 1
[buildozer]
log_level = 2
warn_on_root = 0

[android]
android.api = 31
android.ndk_api = 21
android.archs = arm64-v8a,armeabi-v7a
