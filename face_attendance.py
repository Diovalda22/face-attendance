# import cv2
# import face_recognition
# import pickle
# import datetime
# import pandas as pd

# # Load encoding wajah
# with open("encodings.pkl", "rb") as f:
#     data = pickle.load(f)

# # Inisialisasi kamera
# cap = cv2.VideoCapture(0)

# # File absensi
# attendance_file = "attendance/absensi.csv"

# try:
#     existing_data = pd.read_csv(attendance_file)
# except:
#     existing_data = pd.DataFrame(columns=["Nama", "Waktu"])

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
#         name = "Unknown"

#         if True in matches:
#             matched_idx = matches.index(True)
#             name = data["names"][matched_idx]

#             # Catat absensi jika belum tercatat
#             current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             if name not in existing_data["Nama"].values:
#                 new_entry = pd.DataFrame({"Nama": [name], "Waktu": [current_time]})
#                 existing_data = pd.concat([existing_data, new_entry], ignore_index=True)
#                 existing_data.to_csv(attendance_file, index=False)
#                 print(f"✅ Absensi {name} tercatat!")

#         # Gambar kotak & nama
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow("Sistem Absensi Wajah", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import face_recognition
import pickle
import datetime
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000/api/presensi-face"
CLASS_ID = 3  # Default class ID

# Load face encodings with student names
with open("encodings.pkl", "rb") as f:
    face_data = pickle.load(f)

def send_attendance(student_name):
    payload = {
        'nama_siswa': student_name,
        'kelas_id': CLASS_ID
    }
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        return response.json()
    except Exception as e:
        return {'error': str(e)}

def main():
    cap = cv2.VideoCapture(0)
    last_attendance = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare faces
            matches = face_recognition.compare_faces(face_data["encodings"], face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = face_data["names"][matched_idx]
                
                # Check if we've already processed this person recently
                now = datetime.datetime.now()
                if name not in last_attendance or (now - last_attendance[name]).seconds > 30:
                    # Send to Laravel API
                    result = send_attendance(name)
                    print(f"Attendance result for {name}: {result}")
                    last_attendance[name] = now

            # Display rectangle and name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Face Recognition Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()