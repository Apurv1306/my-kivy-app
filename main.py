from __future__ import annotations

import glob
import json
import os
import queue
import random
import smtplib
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
# Import Window from Kivy for setting orientation explicitly if needed, though buildozer.spec handles most cases
from kivy.core.window import Window


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# SAMPLES_PER_USER is the number of images to capture for each user during registration.
SAMPLES_PER_USER: int = 10
# Factor to reduce frame size for faster face detection.
FRAME_REDUCE_FACTOR: float = 0.5
# Time in seconds before a recognized face can be re-recorded for attendance.
RECOGNITION_INTERVAL: int = 5 * 60  # seconds between repeated recognitions of same face
# Path to the audio file played on successful attendance submission.
AUDIO_FILE: str = "thank_you.mp3"
# Path to the tick mark icon overlay.
TICK_ICON_PATH: str = "tick.png"

# Google-Form configuration: View URL is used as referer header, POST goes to
# the *formResponse* endpoint.
GOOGLE_FORM_VIEW_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
)
GOOGLE_FORM_POST_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
)
FORM_FIELDS: Dict[str, str] = {
    "name": "entry.935510406",
    "emp_id": "entry.886652582",
    "date": "entry.1160275796",
    "time": "entry.32017675",
}

# E-mail (OTP) settings. THESE SHOULD BE PROVIDED VIA ENVIRONMENT VARIABLES
# FOR SECURITY – fallback values are for offline testing only.
EMAIL_ADDRESS: str = os.environ.get("FACEAPP_EMAIL", "faceapp0011@gmail.com")
EMAIL_PASSWORD: str = os.environ.get("FACEAPP_PASS", "ytup bjrd pupf tuuj")
SMTP_SERVER: str = "smtp.gmail.com"
SMTP_PORT: int = 587

# Simple logger helper (replace with logging module for production).
Logger = print

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    """Create directory *path* (including parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def python_time_now() -> str:
    """Returns the current time formatted as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class FaceApp(App):
    """Kivy application for face-recognition based attendance."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        # Set the known faces directory to a writable location on mobile devices.
        # self.user_data_dir is a Kivy property that provides a platform-specific,
        # writable directory for application data. This ensures persistence on Android.
        self._known_faces_dir = Path(self.user_data_dir) / "known_faces"
        ensure_dir(self._known_faces_dir)
        Logger(f"[INFO] Known faces directory set to: {self._known_faces_dir}")

        # Haar cascade for face detection.
        # This XML file is part of OpenCV's data and is used to detect faces.
        # When running on PC, ensure haarcascade_frontalface_default.xml is in the same directory
        # as your Python script, or provide its full path.
        cascade_path = "haarcascade_frontalface_default.xml"
        if not Path(cascade_path).is_file():
            # Fallback to OpenCV's default data path if not found locally, though it's best to include it.
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            Logger(f"[WARN] haarcascade_frontalface_default.xml not found locally. Attempting to load from OpenCV data path: {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Critical check: if the cascade classifier fails to load, face detection won't work.
        if self.face_cascade.empty():
            Logger(f"[ERROR] Failed to load Haar cascade classifier. "
                   f"Path tried: {cascade_path}. "
                   f"Please ensure 'haarcascade_frontalface_default.xml' is present and accessible, "
                   f"and that opencv-python is correctly installed.")
            # Raise a runtime error to halt execution if the cascade is missing/corrupted.
            raise RuntimeError("Failed to load face cascade classifier. Exiting.")


        # Train recogniser on existing samples.
        # This method loads existing face images and trains the LBPHFaceRecognizer.
        self.recognizer, self.label_map = self._train_recognizer()

        # State dictionaries to manage application flow and data.
        self.last_seen_time: Dict[str, float] = {} # Tracks last attendance time for cooldown.
        self.otp_storage: Dict[str, str] = {} # Stores OTPs for verification.
        self.pending_names: Dict[str, Optional[str]] = {} # Stores names during OTP flow.

        # Load stored e-mail addresses for OTP delivery.
        self.user_emails: Dict[str, str] = self._load_emails()

        # Frame queue to pass frames from the camera thread to the UI thread.
        # Max size 1 ensures only the latest frame is kept, reducing lag.
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        # Tick overlay icon (RGBA PNG) – optional visual feedback.
        self.tick_icon: Optional[np.ndarray] = self._load_tick_icon()

        # Optional sound played upon successful attendance recording.
        self.sound = SoundLoader.load(AUDIO_FILE) or None

        # Threading primitives for managing the camera loop.
        self._stop_event = threading.Event() # Used to signal the camera thread to stop.
        self.capture_thread: Optional[threading.Thread] = None # Reference to the camera thread.

        # Attributes for visual flash feedback on image capture.
        self.flash_event = None
        self.flash_rect = None


    # ---------------------------------------------------------------------
    # Kivy UI building / tearing down
    # ---------------------------------------------------------------------

    def build(self):  # noqa: D401 (Kivy signature)
        """Builds the Kivy UI layout."""
        root = FloatLayout()

        # Live camera frame display widget.
        self.image_widget = Image(allow_stretch=True, keep_ratio=True)
        root.add_widget(self.image_widget)

        # Label for displaying status messages (e.g., "Attendance recorded!", "Capturing in 3...")
        self.status_label = Label(
            text="",
            size_hint=(None, None),
            size=(dp(400), dp(50)),
            pos_hint={"center_x": 0.5, "top": 0.95},
            color=(1, 1, 0, 1), # Default color (yellow)
            font_size=dp(20),
            bold=True,
            halign='center',
            valign='middle'
        )
        root.add_widget(self.status_label)

        # Label for displaying current live time (added per user request)
        self.time_label = Label(
            text="",
            size_hint=(None, None),
            size=(dp(200), dp(30)), # Adjust size as needed
            pos_hint={"right": 0.98, "top": 0.98}, # Top-right corner
            color=(1, 1, 1, 1), # White color
            font_size=dp(16),
            bold=True,
            halign='right',
            valign='top'
        )
        root.add_widget(self.time_label)
        # Schedule time label update every second
        Clock.schedule_interval(self._update_time_label, 1)


        # Button bar at the bottom for registration and updates.
        button_bar = BoxLayout(
            orientation="horizontal",
            size_hint=(1, None),
            height=dp(48), # dp ensures density-independent pixels for consistent sizing.
            pos_hint={"center_x": 0.5, "y": 0.02},
            spacing=dp(10),
            padding=dp(10),
        )
        self.register_btn = Button(
            text="Register New Face", background_color=(0.13, 0.59, 0.95, 1) # Blue color
        )
        self.update_btn = Button(
            text="Update Photos", background_color=(0.20, 0.80, 0.20, 1) # Green color
        )
        button_bar.add_widget(self.register_btn)
        button_bar.add_widget(self.update_btn)
        root.add_widget(button_bar)

        # Add stylish borders around buttons for better visual appeal.
        for btn in (self.register_btn, self.update_btn):
            with btn.canvas.after:
                Color(1, 1, 1, 1) # White color for border
                Line(width=1.5, rectangle=(btn.x, btn.y, btn.width, btn.height))
            # Bind to pos and size changes to redraw the border correctly.
            btn.bind(pos=self._update_btn_border, size=self._update_btn_border)

        # Event bindings for button presses.
        self.register_btn.bind(on_press=self._register_popup)
        self.update_btn.bind(on_press=self._update_photos_popup)

        # Open webcam. ALWAYS attempt to open camera index 1 (often front camera for mobile).
        # On PC, camera index 0 is often the default webcam.
        # For PC testing, you might need to try 0 or another index.
        self.capture = cv2.VideoCapture(1) # Attempt to open camera index 1
        if not self.capture.isOpened():
            # If camera 1 fails, log the error and try camera 0 as a common fallback for PC.
            # This specific change is for better PC debugging/testing compatibility.
            Logger("[WARN] Could not open camera index 1. Attempting to open camera index 0 as fallback for PC testing.")
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError("Cannot open any camera (neither index 1 nor 0). Please ensure camera is available and drivers are installed.")
            else:
                Logger("[INFO] Successfully opened camera at index 0.")
        else:
            Logger("[INFO] Successfully opened camera at index 1.")


        # Start a separate thread for camera capture and processing to keep UI responsive.
        self.capture_thread = threading.Thread(
            target=self._camera_loop, daemon=True, name="CameraThread"
        )
        self.capture_thread.start()

        # Schedule UI texture updates at approximately 30 frames per second.
        Clock.schedule_interval(self._update_texture, 1 / 30)


        return root

    def on_stop(self) -> None:  # noqa: D401 (Kivy signature)
        """Called by Kivy when the application is shutting down."""
        # Signal the camera thread to stop gracefully.
        self._stop_event.set()

        # Wait for the camera thread to finish (with a timeout).
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        # Release the camera resource.
        if self.capture:
            self.capture.release()

        Logger(f"[INFO] Application closed cleanly – {python_time_now()}")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_btn_border(instance, *_):  # noqa: ANN001 (Kivy signature)
        """Updates the border around a Kivy button."""
        instance.canvas.after.clear()
        with instance.canvas.after:
            Color(1, 1, 1, 1) # White color for border
            Line(width=1.5, rectangle=(instance.x, instance.y, instance.width, instance.height))

    def _show_popup(self, title: str, content: BoxLayout, *, size=(0.8, 0.5)) -> Popup:  # noqa: D401
        """
        Helper to display a Kivy popup, now with an integrated "Back to Camera" button.
        This ensures users can always return to the main view from any popup.
        """
        # Create a new BoxLayout to hold the original content and the back button
        main_content_layout = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        
        # Add the original content to this new layout
        main_content_layout.add_widget(content)

        # Add a back button at the bottom of the popup
        back_button = Button(
            text="Back to Camera",
            size_hint=(1, None),
            height=dp(40),
            background_color=(0.5, 0.5, 0.5, 1) # Grey color for back button
        )
        main_content_layout.add_widget(back_button)

        # Create the popup with the new content layout
        popup = Popup(title=title, content=main_content_layout, size_hint=size, auto_dismiss=False)
        
        # Bind the back button to dismiss the popup
        back_button.bind(on_press=popup.dismiss)
        
        popup.open()
        return popup

    def _show_status_message(self, message: str, duration: float = 3.0, color=(1, 1, 0, 1)):
        """
        Displays a temporary status message on the UI.
        Args:
            message (str): The message to display.
            duration (float): How long the message should be visible in seconds.
            color (tuple): RGBA color tuple for the message text.
        """
        def update_label(_dt):
            self.status_label.text = message
            self.status_label.color = color
            Clock.schedule_once(lambda __dt: self._clear_status_message(), duration)
        # Schedule the label update on the main Kivy thread
        Clock.schedule_once(update_label, 0)

    def _clear_status_message(self):
        """Clears the status message from the UI."""
        self.status_label.text = ""
        self.status_label.color = (1, 1, 0, 1) # Reset to default color

    def _update_time_label(self, _dt):
        """Updates the live time label with the current time."""
        current_time = datetime.now().strftime("%I:%M:%S %p") # e.g., 01:02:03 PM
        self.time_label.text = current_time

    def _flash_image_widget(self):
        """Briefly flashes a green border around the image widget to indicate a photo capture."""
        # Clear any existing flash event to prevent multiple flashes overlapping
        if self.flash_event:
            self.flash_event.cancel()
            if self.flash_rect:
                self.image_widget.canvas.after.remove(self.flash_rect)

        with self.image_widget.canvas.after:
            Color(0, 1, 0, 1)  # Green color for flash
            # Draw a rectangle that matches the image widget's current size and position
            self.flash_rect = Line(
                width=3,
                rectangle=(
                    self.image_widget.x,
                    self.image_widget.y,
                    self.image_widget.width,
                    self.image_widget.height
                )
            )

        # Schedule clearing the flash after a short duration
        self.flash_event = Clock.schedule_once(self._clear_flash, 0.1)

    def _clear_flash(self, _dt):
        """Clears the green border flash from the image widget."""
        if self.flash_rect:
            self.image_widget.canvas.after.remove(self.flash_rect)
            self.flash_rect = None
        self.flash_event = None # Clear the event reference


    def _load_tick_icon(self) -> Optional[np.ndarray]:
        """Loads the tick icon for overlay, if available.
        Returns None if the file is not found, preventing crashes.
        """
        if not Path(TICK_ICON_PATH).is_file():
            Logger(f"[WARN] Tick icon '{TICK_ICON_PATH}' missing – overlay disabled.")
            return None
        # IMREAD_UNCHANGED ensures alpha channel is read if present.
        return cv2.imread(TICK_ICON_PATH, cv2.IMREAD_UNCHANGED)

    # ------------------------------------------------------------------
    # Camera capture + recognition thread
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        """
        Runs in a background thread: continuously captures frames, detects,
        and recognizes faces. This keeps the UI responsive.
        """
        while not self._stop_event.is_set(): # Loop until stop event is set
            ret, frame = self.capture.read() # Read a frame from the camera
            if not ret:
                Logger("[WARN] Failed to read frame from camera. Retrying...")
                time.sleep(0.1) # Small delay before trying again
                continue  # Skip invalid frames.

            # Down-scale the frame for faster face detection processing.
            h, w = frame.shape[:2]
            resized = cv2.resize(
                frame, (int(w * FRAME_REDUCE_FACTOR), int(h * FRAME_REDUCE_FACTOR))
            )
            gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # Convert to grayscale

            # Perform Haar cascade face detection.
            try:
                faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
            except cv2.error as e:
                Logger(f"[ERROR] OpenCV error in detectMultiScale: {e}. This might indicate a corrupted cascade file or an issue with your OpenCV installation.")
                faces = [] # Treat as no faces detected if an error occurs.


            for (x, y, w_s, h_s) in faces:
                # Scale coordinates back to the original frame size for drawing.
                x_full, y_full, w_full, h_full = [
                    int(v / FRAME_REDUCE_FACTOR) for v in (x, y, w_s, h_s)
                ]

                # Extract the Face Region of Interest (ROI) and convert to grayscale for recognition.
                face_roi = cv2.cvtColor(
                    frame[y_full : y_full + h_full, x_full : x_full + w_full], cv2.COLOR_BGR2GRAY
                )
                try:
                    # Predict the label (person ID) and confidence for the detected face.
                    label, conf = self.recognizer.predict(face_roi)
                except Exception as e:
                    # If prediction fails, treat as unknown. This can happen if recognizer is not trained.
                    Logger(f"[WARN] Face recognition prediction failed: {e}. Treating as unknown.")
                    label, conf = -1, 1000  # Unknown label, high confidence (meaning low match).

                name, emp_id = self.label_map.get(label, ("unknown", ""))
                now = time.time()

                if conf < 60:  # If confidence is below threshold, face is recognized.
                    last_seen = self.last_seen_time.get(emp_id, 0)
                    # Check if enough time has passed since last recognition for this person.
                    if now - last_seen > RECOGNITION_INTERVAL:
                        self.last_seen_time[emp_id] = now # Update last seen time
                        # Submit attendance in a new thread to avoid blocking the camera feed.
                        threading.Thread(
                            target=self._handle_successful_recognition,
                            args=(name, emp_id),
                            daemon=True,
                            name="AttendanceSubmitter",
                        ).start()
                        # Show a success message on the UI.
                        self._show_status_message(f"Attendance recorded for {name.title()}!", 3, (0, 1, 0, 1)) # Green color
                    else:
                        # Inform if attendance was already recently recorded.
                        self._show_status_message(f"Attendance already recorded for {name.title()}.", 3, (1, 0.5, 0, 1)) # Orange color
                    
                    # Draw a green rectangle and text for recognized faces.
                    cv2.rectangle(
                        frame, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"{name.title()} ({emp_id})",
                        (x_full, y_full - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    # Overlay the tick mark next to the name.
                    self._overlay_tick_next_to_name(frame, x_full, y_full - 10, name.title(), emp_id, 0.7, 2)
                else:  # Face is unknown.
                    # Draw a red rectangle and "Unknown" text.
                    cv2.rectangle(
                        frame, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 0, 255), 2
                    )
                    cv2.putText(
                        frame,
                        "Unknown",
                        (x_full, y_full - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # Place the latest processed frame into the queue for UI update.
            # If the queue is full, discard the older frame to ensure low latency.
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    # ------------------------------------------------------------------
    # UI texture refresh (main thread)
    # ------------------------------------------------------------------

    def _update_texture(self, _dt) -> None:  # noqa: D401 (Kivy signature)
        """Updates the Kivy Image widget with the latest camera frame from the queue."""
        if self.frame_queue.empty():
            return
        frame = self.frame_queue.get()
        # Flip the frame vertically and convert to bytes for Kivy texture compatibility.
        # Kivy textures typically expect data in a specific format, and `flip(frame, 0)`
        # is often needed for correct orientation from OpenCV to Kivy.
        buf = cv2.flip(frame, 0).tobytes()
        # Create a Kivy texture and blit the buffer.
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.image_widget.texture = img_texture

    # ------------------------------------------------------------------
    # Training / retraining recogniser
    # ------------------------------------------------------------------

    def _train_recognizer(self):  # noqa: D401 (private helper)
        """Trains the LBPH face recognizer on known faces data."""
        images: list[np.ndarray] = []
        labels: list[int] = []
        label_map: Dict[int, Tuple[str, str]] = {}
        label_id = 0

        # Iterate through files in the known faces directory.
        for file in sorted(os.listdir(self._known_faces_dir)):
            if not file.lower().endswith((".jpg", ".png")):
                continue # Skip non-image files.
            try:
                # Parse name and employee ID from the filename (e.g., "john_doe_EMP001_001.jpg").
                # Expected format: name_empID_NNN.jpg
                parts = file.split("_")
                if len(parts) < 3:
                    Logger(f"[WARN] Skipping file with unexpected filename format: {file}")
                    continue
                name = "_".join(parts[:-2]).lower() # Rejoin parts in case name has underscores
                emp_id = parts[-2].upper() # Employee ID is second to last part
                # The last part is the sample number with extension, which we don't need for name/id
            except ValueError:
                Logger(f"[WARN] Skipping unrecognised filename format: {file}")
                continue

            img_path = self._known_faces_dir / file
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                Logger(f"[WARN] Could not read image: {img_path}. Skipping.")
                continue # Skip if image cannot be read.

            # Resize image to a consistent size (200x200) for training.
            # This is crucial for recognizer performance.
            img_resized = cv2.resize(img_gray, (200, 200))
            images.append(img_resized)
            
            # Assign a new label_id if this emp_id is not already in the map
            # This ensures that each unique emp_id gets a unique integer label
            found_existing_label = False
            for existing_label, (existing_name, existing_emp_id) in label_map.items():
                if existing_emp_id == emp_id:
                    labels.append(existing_label)
                    found_existing_label = True
                    break
            
            if not found_existing_label:
                labels.append(label_id)
                label_map[label_id] = (name, emp_id)
                label_id += 1


        # Create the LBPH face recognizer.
        recogniser = cv2.face.LBPHFaceRecognizer_create()
        if images and labels: # Ensure both lists are not empty
            try:
                # Train the recognizer if there are images available.
                recogniser.train(images, np.array(labels))
                Logger(
                    f"[INFO] Trained recogniser on {len(images)} images across {len(label_map)} identities."
                )
            except cv2.error as e:
                Logger(f"[ERROR] OpenCV error during recognizer training: {e}. This might indicate insufficient samples or corrupted data.")
                recogniser = cv2.face.LBPHFaceRecognizer_create() # Re-initialize empty recognizer
                label_map = {} # Clear label map
        else:
            Logger("[INFO] No images found – recogniser disabled until first registration.")

        return recogniser, label_map

    # ------------------------------------------------------------------
    # Registration / update photo flows
    # ------------------------------------------------------------------

    # -------- Registration --------

    def _register_popup(self, _btn):  # noqa: ANN001 (Kivy signature)
        """Shows the popup for new face registration, collecting user details."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        name_input = TextInput(hint_text="Full Name", size_hint=(1, None), height=dp(40))
        id_input = TextInput(hint_text="Employee ID", size_hint=(1, None), height=dp(40))
        email_input = TextInput(hint_text="Email", size_hint=(1, None), height=dp(40))
        submit_btn = Button(text="Capture Faces", size_hint=(1, None), height=dp(40))

        for widget in (
            Label(text="Enter Details"),
            name_input,
            id_input,
            email_input,
            submit_btn,
        ):
            content.add_widget(widget)
        popup = self._show_popup("Register Face", content, size=(0.9, 0.6))

        def _submit(_):  # noqa: ANN001
            """Handles submission of registration details and starts capture."""
            name = name_input.text.strip().lower().replace(" ", "_")
            emp_id = id_input.text.strip().upper()
            email = email_input.text.strip()
            if not (name and emp_id and email and "@" in email):
                Logger("[WARN] Invalid input for registration.")
                self._show_status_message("Please fill all fields correctly!", 2, (1, 0, 0, 1))
                return

            self._save_email(emp_id, email) # Save email for future OTPs
            popup.dismiss()
            # Start sample capture in a new thread. 'False' indicates new registration.
            threading.Thread(
                target=self._capture_samples,
                args=(name, emp_id, False),
                daemon=True,
                name="CaptureSamples(New)",
            ).start()

        submit_btn.bind(on_press=_submit)

    # -------- Update Existing Photos --------

    def _update_photos_popup(self, _btn):  # noqa: ANN001
        """Shows the popup for updating existing user photos, starting with EMP ID."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        content.add_widget(Label(text="Enter your Employee ID:"))
        emp_input = TextInput(hint_text="EMP ID", size_hint=(1, None), height=dp(40))
        next_btn = Button(text="Next", size_hint=(1, None), height=dp(40))
        for w in (emp_input, next_btn):
            content.add_widget(w)
        popup = self._show_popup("Update Photos", content)

        def _next(_):  # noqa: ANN001
            """Handles the next step in the update photos flow (OTP or email registration)."""
            emp_id = emp_input.text.strip().upper()
            if not emp_id:
                Logger("[WARN] Employee ID cannot be empty for update.")
                self._show_status_message("Employee ID cannot be empty!", 2, (1, 0, 0, 1))
                return
            email = self.user_emails.get(emp_id)
            name_existing: Optional[str] = None
            # Find the existing name associated with the EMP ID for consistent filename.
            for _lbl, (nm, eid) in self.label_map.items():
                if eid == emp_id:
                    name_existing = nm
                    break
            
            if name_existing is None:
                Logger(f"[WARN] Employee ID {emp_id} not found in known faces. Cannot update.")
                self._show_status_message("Employee ID not found. Please register first.", 3, (1, 0, 0, 1))
                popup.dismiss()
                return

            popup.dismiss()
            if email:
                # If email exists, proceed to send OTP.
                self._send_otp_flow(emp_id, email, name_existing)
            else:
                # If no email, prompt for email registration first.
                self._email_registration_flow(emp_id, name_existing)

        next_btn.bind(on_press=_next)

    # ------------------------------------------------------------------
    # Helper flows for OTP / email registration
    # ------------------------------------------------------------------

    def _email_registration_flow(self, emp_id: str, name: Optional[str]):
        """Initiates the flow to register an email for an existing employee ID."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        content.add_widget(Label(text="Email not found for this ID. Please enter your email:"))
        email_input = TextInput(hint_text="Email", size_hint=(1, None), height=dp(40))
        submit_btn = Button(text="Submit", size_hint=(1, None), height=dp(40))
        content.add_widget(email_input)
        content.add_widget(submit_btn)
        popup = self._show_popup("Register Email", content)

        def _submit(_):  # noqa: ANN001
            """Handles email submission for registration."""
            email = email_input.text.strip()
            if email and "@" in email:
                self._save_email(emp_id, email) # Save the new email.
                popup.dismiss()
                self._send_otp_flow(emp_id, email, name) # Proceed to OTP.
            else:
                Logger("[WARN] Invalid email format during registration.")
                self._show_status_message("Invalid email format!", 2, (1, 0, 0, 1))


        submit_btn.bind(on_press=_submit)

    def _send_otp_flow(self, emp_id: str, email: str, name: Optional[str]):
        """Manages the OTP sending process, showing a sending popup."""
        # Generate & store 6-digit OTP.
        otp = self._generate_otp()
        self.otp_storage[emp_id] = otp
        self.pending_names[emp_id] = name # Store name temporarily for capture after OTP.

        # Use a temporary label for the sending message
        sending_message_label = Label(text="Sending OTP email…", size_hint=(1, 1), halign='center', valign='middle')
        sending_popup = self._show_popup("Sending OTP", BoxLayout(children=[sending_message_label]), size=(0.7, 0.4))

        def _send_thread():  # noqa: ANN001
            """Sends the OTP email in a separate thread to avoid freezing UI."""
            ok = self._send_otp_email(email, otp)
            Clock.schedule_once(lambda _dt: sending_popup.dismiss()) # Dismiss sending popup on main thread.
            if ok:
                Clock.schedule_once(lambda _dt: self._otp_verify_popup(emp_id, email)) # Show verification popup.
            else:
                Clock.schedule_once(
                    lambda _dt: self._show_popup("Error", Label(text="Failed to send email. Please check console for details.\nEnsure app is allowed in Google security settings."), size=(0.7, 0.4))
                )

        threading.Thread(target=_send_thread, daemon=True, name="SendOTPThread").start()

    def _otp_verify_popup(self, emp_id: str, email: str):
        """Shows the popup for OTP verification."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        content.add_widget(Label(text=f"OTP sent to {email}\nEnter the 6-digit OTP:"))
        otp_input = TextInput(hint_text="6-digit OTP", input_type="number", size_hint=(1, None), height=dp(40)) # Input type for numbers
        verify_btn = Button(text="Verify", size_hint=(1, None), height=dp(40))
        resend_btn = Button(text="Resend", size_hint=(1, None), height=dp(40))
        content.add_widget(otp_input)
        content.add_widget(verify_btn)
        content.add_widget(resend_btn)
        popup = self._show_popup("Verify OTP", content)

        def _verify(_):  # noqa: ANN001
            """Verifies the entered OTP."""
            if otp_input.text.strip() == self.otp_storage.get(emp_id):
                popup.dismiss()
                name_for_capture = self.pending_names.get(emp_id)
                # Start sample capture for update. 'True' indicates update, '5' samples.
                threading.Thread(
                    target=self._capture_samples,
                    args=(name_for_capture, emp_id, True, 5),
                    daemon=True,
                    name="CaptureSamples(Update)",
                ).start()
            else:
                otp_input.text = ""
                otp_input.hint_text = "Incorrect – try again"
                self._show_status_message("Incorrect OTP. Try again!", 2, (1, 0, 0, 1))
                Logger("[WARN] Incorrect OTP entered.")

        def _resend(_):  # noqa: ANN001
            """Resends a new OTP."""
            # Clear current OTP and pending name, then restart the OTP flow.
            if emp_id in self.otp_storage:
                del self.otp_storage[emp_id]
            if emp_id in self.pending_names:
                del self.pending_names[emp_id]
            
            popup.dismiss()
            self._show_status_message("Resending OTP...", 2, (1, 1, 0, 1))
            # Restart the send OTP flow to regenerate and send new OTP
            self._send_otp_flow(emp_id, email, self.pending_names.get(emp_id)) # Use pending name if it exists

        verify_btn.bind(on_press=_verify)
        resend_btn.bind(on_press=_resend)

    # ------------------------------------------------------------------
    # Core logic: capturing face samples, attendance submission
    # ------------------------------------------------------------------

    def _capture_samples(
        self,
        name: Optional[str],
        emp_id: str,
        updating: bool = False,
        sample_count: Optional[int] = None,
    ):
        """
        Captures face samples for a given user.
        Handles both new registrations (10 samples) and updates (5 samples).
        Provides visual countdown and feedback.
        """
        # Resolve name for existing employee ID if not supplied (e.g., during update flow).
        if name is None:
            # Iterate through label_map to find the name associated with emp_id
            for _lbl, (nm, eid) in self.label_map.items():
                if eid == emp_id:
                    name = nm
                    break
        if name is None:
            Logger("[ERROR] No name found for this Employee ID. This should not happen in update flow if checks pass.")
            Clock.schedule_once(lambda _dt: self._show_popup("Error", Label(text="Internal error: Employee name not found for capture."), size=(0.7, 0.4)))
            return # Exit if name is still None

        # Determine target number of samples (10 for new, 5 for update).
        count_target = sample_count if sample_count else SAMPLES_PER_USER
        # Construct pattern to find existing files for this user.
        pattern = str(self._known_faces_dir / f"{name}_{emp_id}_*.jpg")
        existing_files = glob.glob(pattern)
        start_index = len(existing_files) # Start numbering new files from here.
        collected = 0 # Counter for successfully collected samples.

        Logger(
            f"[INFO] Starting sample capture for {emp_id} – target {count_target} faces (updating={updating})."
        )

        # --- Visual Countdown before capture starts ---
        for i in range(3, 0, -1):
            # Schedule on the main thread for UI update
            Clock.schedule_once(lambda _dt, msg=f"Capturing in {i}...": self._show_status_message(msg, 1, (1, 1, 0, 1)), 0)
            time.sleep(1) # Pause for 1 second.
        Clock.schedule_once(lambda _dt: self._show_status_message("Capturing now!", 1, (0, 1, 0, 1)), 0) # Green "Capturing now!".
        time.sleep(0.5) # Small pause before actual capture loop.

        # Loop to capture the target number of samples.
        while collected < count_target and not self._stop_event.is_set():
            # Get the latest frame from the queue. Non-blocking to keep camera thread flowing.
            frame = None
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                pass # No frame available yet

            if frame is None:
                time.sleep(0.01) # Small sleep if no frame available to prevent busy-waiting.
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the current frame.
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0: # If at least one face is detected.
                # Take the first detected face (assuming one person is registering).
                x, y, w, h = faces[0] 
                face_img = gray[y : y + h, x : x + w]
                # Check if face_img is empty or too small (e.g., if detection was faulty)
                if face_img.size == 0 or w < 50 or h < 50: # Minimum size for a valid face
                    self._show_status_message("Face too small or incomplete. Adjust position.", 0.5, (1, 0, 0, 1))
                    time.sleep(0.1)
                    continue

                # Resize the captured face image to 200x200 pixels.
                face_img_resized = cv2.resize(face_img, (200, 200))
                # Construct a unique filename for the captured image.
                filename = f"{name}_{emp_id}_{start_index + collected:03d}.jpg"
                # Save the resized image to the internal storage directory.
                cv2.imwrite(str(self._known_faces_dir / filename), face_img_resized)
                collected += 1 # Increment collected count.
                Logger(f"[INFO] Captured sample {collected}/{count_target} for {emp_id}")
                
                # Provide visual feedback on the UI that a photo was captured.
                Clock.schedule_once(lambda _dt: self._flash_image_widget(), 0)

                # Update status message with capture progress.
                Clock.schedule_once(lambda _dt, msg=f"Captured {collected}/{count_target} photos...": self._show_status_message(msg, 0.5, (1, 1, 0, 1)), 0)
                # Small delay to allow for head movement between samples for better training data.
                time.sleep(0.2)
            else:
                # If no face is detected, prompt the user to adjust position.
                Clock.schedule_once(lambda _dt: self._show_status_message("No face detected. Please position yourself.", 0.5, (1, 0, 0, 1)), 0) # Red warning.
                time.sleep(0.1) # Small delay.


        Logger("[INFO] Capture complete – retraining recogniser…")
        # Retrain the recognizer with the new/updated samples.
        self.recognizer, self.label_map = self._train_recognizer()
        Logger("[INFO] Update finished.")
        # Display final completion message based on whether it was a registration or update.
        if updating:
            Clock.schedule_once(lambda _dt: self._show_status_message("Face updated!", 3, (0, 1, 0, 1)), 0)
        else:
            Clock.schedule_once(lambda _dt: self._show_status_message("Registration completed!", 3, (0, 1, 0, 1)), 0)


    # ------------------------------------------------------------------
    # Successful recognition & attendance submission
    # ------------------------------------------------------------------

    def _handle_successful_recognition(self, name: str, emp_id: str):
        """Handles a successful face recognition event, playing sound and submitting attendance."""
        Logger(f"[INFO] Recognised {name} ({emp_id}) – submitting attendance…")
        if self.sound:
            try:
                self.sound.play() # Play success sound.
            except Exception as e:
                Logger(f"[WARN] Failed to play sound: {e}")

        # Submit to Google Form in a new thread to prevent UI freezing.
        threading.Thread(
            target=self._submit_to_google_form,
            args=(name, emp_id),
            daemon=True,
            name="GoogleFormSubmitter",
        ).start()

    def _submit_to_google_form(self, name: str, emp_id: str) -> None:
        """
        Submits attendance data to a Google Form.
        Includes robust error handling for network issues and form response.
        """
        payload = {
            FORM_FIELDS["name"]: name.title(),
            FORM_FIELDS["emp_id"]: emp_id,
            FORM_FIELDS["date"]: datetime.now().strftime("%d/%m/%Y"),
            FORM_FIELDS["time"]: datetime.now().strftime("%H:%M:%S"),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (FaceApp Attendance Bot)",
            "Referer": GOOGLE_FORM_VIEW_URL,
        }
        
        Logger(f"[INFO] Attempting to submit attendance for {name} ({emp_id}) to URL: {GOOGLE_FORM_POST_URL}")
        Logger(f"[INFO] Payload: {payload}")
        try:
            with requests.Session() as session:
                resp = session.post(
                    GOOGLE_FORM_POST_URL,
                    data=payload,
                    headers=headers,
                    timeout=10, # Set a timeout for the request.
                    allow_redirects=False, # Google Forms typically redirects on success (302).
                )
            
            # Check for successful status codes (200 OK or 302 Found for redirect).
            if resp.status_code in (200, 302):
                Logger("[INFO] Attendance submitted successfully to Google Form.")
                Clock.schedule_once(lambda _dt: self._show_status_message(f"Attendance submitted for {name.title()}!", 3, (0, 1, 0, 1)), 0)
            else:
                Logger(
                    f"[WARN] Google Form submission returned status {resp.status_code}. "
                    f"Response: {resp.text[:200]}..." # Log part of the response for debugging.
                )
                Clock.schedule_once(lambda _dt: self._show_popup("Submission Warning", Label(text=f"Form submission failed (Status: {resp.status_code}). Please check console for details and verify form configuration."), size=(0.8, 0.5)))
        except requests.exceptions.Timeout:
            Logger(f"[ERROR] Google Form submission timed out for {name} ({emp_id}).")
            Clock.schedule_once(lambda _dt: self._show_popup("Submission Error", Label(text="Form submission timed out. Check network connection."), size=(0.8, 0.5)))
        except requests.exceptions.ConnectionError as exc:
            Logger(f"[ERROR] Google Form submission connection error for {name} ({emp_id}): {exc}")
            Clock.schedule_once(lambda _dt: self._show_popup("Submission Error", Label(text="Network error during form submission. Check internet connection."), size=(0.8, 0.5)))
        except requests.RequestException as exc:
            Logger(f"[ERROR] An unexpected error occurred during form submission for {name} ({emp_id}): {exc}")
            Clock.schedule_once(lambda _dt: self._show_popup("Submission Error", Label(text=f"An error occurred during form submission: {exc}"), size=(0.8, 0.5)))


    # ------------------------------------------------------------------
    # OTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_otp() -> str:
        """Generates a 6-digit random OTP."""
        return str(random.randint(100000, 999999))

    def _send_otp_email(self, email: str, otp: str) -> bool:
        """Sends an OTP email to the specified address."""
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = email
        msg["Subject"] = "Your FaceApp OTP"
        body_html = (
            f"<h2>OTP Verification</h2><p>Your OTP is <b>{otp}</b>. "
            "It is valid for 10 minutes.</p>"
        )
        msg.attach(MIMEText(body_html, "html"))
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls() # Enable Transport Layer Security
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD) # Log in to SMTP server
                server.send_message(msg) # Send the email
            Logger(f"[INFO] Sent OTP to {email}")
            return True
        except smtplib.SMTPAuthenticationError as exc:
            Logger(f"[ERROR] SMTP authentication error: {exc}. "
                   "Check EMAIL_ADDRESS and EMAIL_PASSWORD (especially for App Passwords for Gmail).")
            return False
        except smtplib.SMTPConnectError as exc:
            Logger(f"[ERROR] SMTP connection error: {exc}. "
                   "Check SMTP_SERVER and SMTP_PORT, and network connectivity.")
            return False
        except Exception as exc:
            Logger(f"[ERROR] General SMTP error when sending OTP to {email}: {exc}")
            return False

    # ------------------------------------------------------------------
    # E-mail persistence helpers
    # ------------------------------------------------------------------

    def _load_emails(self) -> Dict[str, str]:
        """Loads stored user email addresses from a JSON file."""
        emails_file = self._known_faces_dir / "user_emails.json"
        if emails_file.is_file():
            try:
                with emails_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                Logger(f"[WARN] Invalid JSON in email storage: {exc}; starting fresh.")
                # It's good practice to back up or rename the corrupted file
                # to prevent continuous errors if it's indeed corrupted.
                # For this example, we'll just log and return empty.
        return {}

    def _save_email(self, emp_id: str, email: str) -> None:
        """Saves a user's email address to the JSON file."""
        self.user_emails[emp_id] = email
        try:
            with (self._known_faces_dir / "user_emails.json").open("w", encoding="utf-8") as f:
                json.dump(self.user_emails, f, indent=2)
        except IOError as exc:
            Logger(f"[ERROR] Failed to save user emails to file: {exc}")

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _overlay_tick_next_to_name(self, frame: np.ndarray, text_x: int, text_y_baseline: int, name: str, emp_id: str, font_scale: float, font_thickness: int) -> None:
        """Overlays a tick icon next to the recognized name and ID."""
        if self.tick_icon is None:
            return

        text_to_measure = f"{name} ({emp_id})"
        (text_width, text_height), _ = cv2.getTextSize(text_to_measure, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Desired size for the tick icon (e.g., 25x25 pixels)
        tick_icon_size = 25 # pixels
        try:
            icon = cv2.resize(self.tick_icon, (tick_icon_size, tick_icon_size), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            Logger(f"[ERROR] Failed to resize tick icon: {e}. Skipping overlay.")
            return

        # Calculate position for the tick mark
        # Place it slightly to the right of the text, vertically centered with the text.
        padding_x = 5 # pixels between text and tick
        
        icon_x_start = text_x + text_width + padding_x
        # Calculate y-position to center the tick vertically with the text
        # text_y_baseline is the bottom of the text. Top of text is text_y_baseline - text_height.
        # Center of text is text_y_baseline - text_height / 2.
        # Center of icon should be at text_center_y. So, icon_y_top = text_center_y - tick_icon_size / 2.
        icon_y_start = text_y_baseline - text_height + (text_height - tick_icon_size) // 2

        # Ensure the icon is within frame boundaries
        h_frame, w_frame = frame.shape[:2]
        icon_x_start = max(0, min(icon_x_start, w_frame - tick_icon_size))
        icon_y_start = max(0, min(icon_y_start, h_frame - tick_icon_size))

        # Ensure icon_x_start and icon_y_start are integers
        icon_x_start = int(icon_x_start)
        icon_y_start = int(icon_y_start)

        # Get the ROI for blending
        roi = frame[icon_y_start : icon_y_start + tick_icon_size, icon_x_start : icon_x_start + tick_icon_size]
        
        # Check if the ROI is valid (i.e., not out of bounds causing a slice of different size)
        if roi.shape[0] == tick_icon_size and roi.shape[1] == tick_icon_size:
            if icon.shape[2] == 4:  # RGBA image – use alpha channel for blending
                # Split icon into B, G, R, Alpha channels
                b, g, r, a = cv2.split(icon)
                # Create a 3-channel mask from the alpha channel
                mask = cv2.merge((a, a, a)) / 255.0
                # Blend the ROI with the icon using the mask
                blended = (roi * (1 - mask) + cv2.merge((b, g, r)) * mask).astype(np.uint8)
                frame[icon_y_start : icon_y_start + tick_icon_size, icon_x_start : icon_x_start + tick_icon_size] = blended
            else:
                Logger("[WARN] Tick icon is not RGBA; cannot perform alpha blending for tick next to name. Simple copy used.")
                # Fallback: simple copy if no alpha channel. Ensure it's BGR if original is not.
                icon_to_place = cv2.cvtColor(icon, cv2.COLOR_BGRA2BGR) if icon.shape[2] == 4 else icon
                frame[icon_y_start : icon_y_start + tick_icon_size, icon_x_start : icon_x_start + tick_icon_size] = icon_to_place
        else:
            Logger(f"[WARN] ROI for tick icon is not the expected size. Skipping overlay. ROI shape: {roi.shape}")


# ---------------------------------------------------------------------------
# Run the application
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        FaceApp().run()
    except Exception as e:
        # This will catch any unhandled exceptions during Kivy app startup or main loop
        # and print them to the console, which is crucial for debugging crashes on PC.
        import traceback
        Logger(f"[CRITICAL ERROR] Application crashed: {e}")
        Logger("--- Full Traceback ---")
        traceback.print_exc()
        Logger("----------------------")
        # You might want to display a simple error popup here too for the user,
        # but for debugging, console output is primary.
