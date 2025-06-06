from flask import Flask, request, render_template
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
DATASET_PATH = "dataset/"

@app.route("/")  # Tambahkan route ini
def index():
    return render_template("index.html")  # Tampilkan halaman HTML

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    image_data = data.get("image").split(",")[1]
    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    path = os.path.join(DATASET_PATH, f"{name}.jpg")
    cv2.imwrite(path, img)
    return f"✅ Wajah {name} berhasil disimpan!"

if __name__ == "__main__":
    app.run(debug=True)
