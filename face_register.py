import cv2
import os
import face_recognition

def register_face(name, dataset_path="dataset/"):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Tekan 'S' untuk Simpan", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Registrasi Wajah - Tekan 'S' untuk Simpan", frame)

        key = cv2.waitKey(1)
        if key == ord('s') and len(faces) > 0:
            face_roi = frame[y:y+h, x:x+w]
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            cv2.imwrite(f"{dataset_path}{name}.jpg", face_roi)
            print(f"✅ Wajah {name} berhasil disimpan!")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Masukkan nama: ")
    register_face(name)