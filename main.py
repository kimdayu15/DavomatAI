from datetime import datetime, timedelta
import face_recognition
import numpy as np
import keyboard
import logging
import math
import cv2
import sys
import csv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance = {}
        self.detection_times = {}
        self.session_start_time = None
        self.process_this_frame = True
        self.stop_recognition = False

        self.late_threshold_minutes = 10  # Threshold for being late in minutes
        self.present_threshold_percentage = 70  # Threshold for being present in percentage

        self.encode_faces()
        keyboard.add_hotkey('q', self.stop_recognition_key)

    def face_confidence(self, face_distance, face_match_threshold=0.6):
        range_val = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)

        if face_distance > face_match_threshold:
            return f'{round(linear_val * 100, 2)}%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return f'{round(value, 2)}%'

    def encode_faces(self):
        logging.info("Encoding faces...")
        for image in os.listdir('faces'):
            try:
                face_image = face_recognition.load_image_file(f'faces/{image}')
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image.split('.')[0])
            except Exception as e:
                logging.warning(f"Failed to encode {image}: {e}")

        logging.info(f"Encoded faces: {self.known_face_names}")

    def mark_attendance(self, name):
        now = datetime.now()
        if name not in self.attendance:
            self.attendance[name] = [now, now]
            self.detection_times[name] = timedelta()
        else:
            self.attendance[name][1] = now

    def update_detection_times(self, names):
        for name in names:
            if name in self.detection_times:
                self.detection_times[name] += timedelta(seconds=1)

    def run_recognition(self):
        logging.info("Starting video capture...")
        video_capture = cv2.VideoCapture(1)
        # video_capture.open('http://192.168.5.67:8080/video')
        # video_capture.open('classroom.JPG')

        if not video_capture.isOpened():
            logging.error('Video source not found...')
            sys.exit('Video source not found...')

        self.session_start_time = datetime.now()

        while not self.stop_recognition:
            ret, frame = video_capture.read()
            if not ret:
                continue

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Process every other frame
            if self.process_this_frame:
                self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=3)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = self.face_confidence(face_distances[best_match_index])
                        self.mark_attendance(name)
                        logging.info(f'Detected {name}')

                    self.face_names.append(f'{name} ({confidence})')

                self.update_detection_times([name.split()[0] for name in self.face_names if name.split()[0] != 'Unknown'])

            self.process_this_frame = not self.process_this_frame

            # Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)
            # cv2.imwrite('recognized_faces.png', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.generate_attendance_report()
        video_capture.release()
        cv2.destroyAllWindows()

    def stop_recognition_key(self):
        logging.info("Stopping face recognition...")
        self.stop_recognition = True

    def generate_attendance_report(self):
        session_end_time = datetime.now()
        session_duration = session_end_time - self.session_start_time
        logging.info("Generating attendance report...")

        with open("attendance_report.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "First Seen", "Last Seen", "Presence Percentage", "Status"])
            for name in self.known_face_names:
                if name in self.attendance:
                    times = self.attendance[name]
                    presence_time = min(self.detection_times[name].total_seconds(), session_duration.total_seconds())
                    presence_percentage = (presence_time / session_duration.total_seconds()) * 100
                    if presence_percentage >= self.present_threshold_percentage:
                        status = "Present"
                    else:
                        first_seen = times[0]
                        if (first_seen - self.session_start_time).total_seconds() > self.late_threshold_minutes * 60:
                            status = "Late"
                        else:
                            status = "Absent"
                    writer.writerow([name, times[0], times[1], f"{presence_percentage:.2f}%", status])
                else:
                    writer.writerow([name, "N/A", "N/A", "0.00%", "Absent"])

        logging.info("Attendance report generated successfully.")


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
