name: Build APK with Buildozer

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y python3-pip cython3 git zip unzip openjdk-17-jdk
          python3 -m pip install --upgrade pip
          pip install cython buildozer

      - name: Build APK
        run: |
          buildozer android debug

      - name: Upload APK
        uses: actions/upload-artifact@v4
        with:
          name: facialrecognition-debug-apk
          path: bin/*.apk
