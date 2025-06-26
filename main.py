import os
import cv2
import numpy as np
import threading
import requests
from kivy import kivymd
from datetime import datetime
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.core.audio import SoundLoader
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

KNOWN_FACES_DIR = "known_faces"
RECOGNITION_INTERVAL = 600
AUDIO_FILE = "thank_you.mp3"
GOOGLE_FORM_URL = "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
FORM_FIELDS = {
    "name": "entry.935510406",
    "emp_id": "entry.886652582",
    "date": "entry.1160275796",
    "time": "entry.32017675",
}

class FaceApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer, self.label_map = self.train_recognizer()
        self.last_seen_time = {}

        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.button = Button(text='Register New Face', size_hint=(1, 0.1))
        self.button.bind(on_press=self.register_popup)

        layout.add_widget(self.image)
        layout.add_widget(self.button)

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if self.recognizer:
            for (x, y, w, h) in faces:
                roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label_id, confidence = self.recognizer.predict(roi)

                if confidence < 70 and label_id in self.label_map:
                    label = self.label_map[label_id]
                    name, emp_id = label.split("_")

                    if name and emp_id:
                        now = time.time()
                        if label_id not in self.last_seen_time or now - self.last_seen_time[label_id] > RECOGNITION_INTERVAL:
                            threading.Thread(target=self.play_sound_and_submit, args=(name, emp_id)).start()
                            self.last_seen_time[label_id] = now

                    cv2.putText(frame, name.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def train_recognizer(self):
        faces, labels = [], []
        label_map = {}
        label_counter = 0
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)

        for file in os.listdir(KNOWN_FACES_DIR):
            if file.endswith((".jpg", ".png")):
                try:
                    name, emp_id, _ = file.split("_")
                    label = f"{name}_{emp_id}"
                except:
                    continue
                img_path = os.path.join(KNOWN_FACES_DIR, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                if label not in label_map.values():
                    label_map[label_counter] = label
                    label_counter += 1
                faces.append(img)
                labels.append(list(label_map.values()).index(label))

        if len(faces) < 2:
            return None, None

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        return recognizer, label_map

    def register_popup(self, instance):
        content = BoxLayout(orientation='vertical')
        name_input = TextInput(hint_text='Full Name')
        id_input = TextInput(hint_text='Employee ID')
        submit_btn = Button(text='Capture Faces')

        content.add_widget(Label(text="Enter Details"))
        content.add_widget(name_input)
        content.add_widget(id_input)
        content.add_widget(submit_btn)

        popup = Popup(title='Register Face', content=content, size_hint=(0.9, 0.6))

        def submit(instance):
            name = name_input.text.strip().lower()
            emp_id = id_input.text.strip().upper()
            popup.dismiss()
            threading.Thread(target=self.register_new_face, args=(name, emp_id)).start()

        submit_btn.bind(on_press=submit)
        popup.open()

    def register_new_face(self, name, emp_id):
        count = 0
        for i in range(1, 11):
            ret, frame = self.capture.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                filename = f"{name}_{emp_id}_{i}.jpg"
                cv2.imwrite(os.path.join(KNOWN_FACES_DIR, filename), face)
                count += 1
                break
            time.sleep(0.2)
        print(f"[INFO] Saved {count} images.")
        self.recognizer, self.label_map = self.train_recognizer()

    def play_sound_and_submit(self, name, emp_id):
        sound = SoundLoader.load(AUDIO_FILE)
        if sound:
            sound.play()
        self.send_to_google_form(name, emp_id)

    def send_to_google_form(self, name, emp_id):
        if not name or not emp_id:
            print("[WARNING] Skipped submission: name or emp_id is empty.")
            return

        now = datetime.now()
        data = {
            FORM_FIELDS["name"]: name,
            FORM_FIELDS["emp_id"]: emp_id,
            FORM_FIELDS["date"]: now.strftime("%Y-%m-%d"),
            FORM_FIELDS["time"]: now.strftime("%H:%M:%S"),
        }
        try:
            requests.post(GOOGLE_FORM_URL, data=data)
            print(f"[INFO] Attendance submitted for {name} ({emp_id})")
        except:
            print("[ERROR] Submission failed.")

if __name__ == '__main__':
    FaceApp().run()
