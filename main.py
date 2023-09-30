import face_recognition
import cv2
import os
import json
from datetime import datetime

member_dir = "member"

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

know_face_encoding = []
know_face_names = []

for member in os.listdir(member_dir):
    member_path = os.path.join("member", member)
    for image_file in os.listdir(member_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(member_path, image_file)
            member_image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(member_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(member_image, cv2.COLOR_BGR2RGB)
            faces = face_detector.detectMultiScale(gray_image)
            for (x,y,w,h) in faces:
                face_location = (y, x+w, y+h,x)
                face_encoding = face_recognition.face_encodings(rgb_image,[face_location])[0]
                know_face_encoding.append(face_encoding)
                know_face_names.append(member)

capture = cv2.VideoCapture(0)

detection_data = []

while True: 
    ret, frame = capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detectMultiScale(gray)
        face_location = [(y, x+w, y+h,x) for (x,y,w,h) in faces]
        face_encoding = face_recognition.face_encodings(rgb, face_location)
        for (top, right, bottom, left), face_encoding in zip(face_location, face_encoding):
            match = face_recognition.compare_faces(know_face_encoding, face_encoding)
            if True in match:
                first_match_index = match.index(True)
                name = know_face_names[first_match_index]
            else:
                name = "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, top-6), font, 0.5, (255, 255, 255),1)

            # Add detection data to the list
            detection_data.append({"name": name, "timestamp": str(datetime.now())})

        cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        with open('detection_data.json', 'w') as json_file:
            json.dump(detection_data, json_file, indent=4)
        break

capture.release()
cv2.destroyAllWindows()
