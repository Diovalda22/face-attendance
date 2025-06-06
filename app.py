from flask import Flask, render_template, request, jsonify
import os, cv2, base64, pickle, datetime
import numpy as np
import pandas as pd
import face_recognition

app = Flask(__name__)

DATASET_DIR = "dataset"
ENCODING_FILE = "encodings.pkl"
ABSEN_FILE = "attendance/absensi.csv"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("attendance", exist_ok=True)

# Load encoding wajah
def load_encodings():
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

# Save encoding wajah
def save_encodings(data):
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)

# Training ulang
def train_faces():
    encodings, names = [], []
    for filename in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, filename)
        img = face_recognition.load_image_file(path)
        faces = face_recognition.face_encodings(img)
        if faces:
            encodings.append(faces[0])
            names.append(os.path.splitext(filename)[0])
    save_encodings({"encodings": encodings, "names": names})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

@app.route("/register", methods=["POST"])
def register():
    req = request.get_json()
    name, img_data = req["name"], req["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return jsonify({"message": "❌ Wajah tidak terdeteksi."})

    filename = os.path.join(DATASET_DIR, f"{name}.jpg")
    cv2.imwrite(filename, frame)

    train_faces()
    return jsonify({"message": f"✅ Wajah {name} berhasil didaftarkan!"})

@app.route("/attendance", methods=["POST"])
def attendance():
    req = request.get_json()
    img_data = req["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    data = load_encodings()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if not face_encodings:
        return jsonify({"message": "❌ Wajah tidak terdeteksi."})

    try:
        absensi = pd.read_csv(ABSEN_FILE)
    except:
        absensi = pd.DataFrame(columns=["Nama", "Waktu"])

    waktu = datetime.datetime.now().strftime("%Y-%m-%d")
    messages = {"success": [], "info": [], "error": []}

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
        if True in matches:
            idx = matches.index(True)
            name = data["names"][idx]

            # Cek apakah sudah absen hari ini
            if not ((absensi["Nama"] == name) & (absensi["Waktu"].str.startswith(waktu))).any():
                waktu_lengkap = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                absensi = pd.concat([absensi, pd.DataFrame({"Nama": [name], "Waktu": [waktu_lengkap]})], ignore_index=True)
                messages["success"].append(f"✅ Absensi {name} berhasil!")
            else:
                messages["info"].append(f"ℹ️ {name} sudah absen hari ini.")
        else:
            messages["error"].append("❌ Wajah tidak dikenal.")

    absensi.to_csv(ABSEN_FILE, index=False)

    # Gabungkan pesan
    result_messages = []
    result_messages.extend(messages["success"])
    result_messages.extend(messages["info"])
    # Untuk error, supaya tidak terlalu banyak, tampilkan sekali saja jika ada error wajah tidak dikenal
    if messages["error"]:
        result_messages.append("❌ Beberapa wajah tidak dikenal.")

    return jsonify({"message": "\n".join(result_messages)})


if __name__ == "__main__":
    app.run(debug=True)
