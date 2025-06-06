import os
import cv2
import face_recognition
import pickle

def train_faces(dataset_path, output_file="encodings.pkl"):
    known_encodings = []
    known_names = []

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(image_name)[0])

    data = {"encodings": known_encodings, "names": known_names}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Training selesai! Data disimpan di {output_file}")

if __name__ == "__main__":
    train_faces("dataset/")