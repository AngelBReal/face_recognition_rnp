from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import pickle
from keras_facenet import FaceNet
import mediapipe as mp

app = Flask(__name__, static_folder='static', static_url_path='/static')
socketio = SocketIO(app)

# Load FaceNet + embeddings
embedder = FaceNet()
with open('face_database.pkl', 'rb') as f:
    database = pickle.load(f)

# Mediapipe face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def recognize_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    embedding = embedder.embeddings([face_img])[0]
    min_dist = 100
    identity = 'desconocido'
    for name, db_embeddings in database.items():
        for db_emb in db_embeddings:
            dist = np.linalg.norm(embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name
    if min_dist > 0.7:
        identity = 'desconocido'
    return identity

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (320, 240))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    identities = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
            face_img = rgb_frame[y1:y2, x1:x2]
            identity = recognize_face(face_img)
            if identity not in identities:
                identities.append(identity)

    if not identities:
        identities.append("sin rostro")

    emit('response', {'identities': identities})

if __name__ == '__main__':
    import eventlet
    eventlet.monkey_patch()
    socketio.run(app)
