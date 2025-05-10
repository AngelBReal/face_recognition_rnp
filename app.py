from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import pickle
from keras_facenet import FaceNet
import mediapipe as mp
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'tu_clave_secreta'

# Configuración de CORS para Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# Load FaceNet + embeddings
embedder = FaceNet()
with open('face_database.pkl', 'rb') as f:
    database = pickle.load(f)

# Mediapipe face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def recognize_face(face_img):
    # Verificar que tenemos una imagen válida
    if face_img is None or face_img.size == 0:
        return 'desconocido'
    try:
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
    except Exception as e:
        print(f"Error en reconocimiento facial: {e}")
        return 'error'

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

@socketio.on('frame')
def handle_frame(data):
    try:
        # Asegúrate de que data es una cadena y tiene el formato correcto
        if not isinstance(data, str) or ',' not in data:
            emit('response', {'identities': ['error formato']})
            return
            
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('response', {'identities': ['error imagen']})
            return
            
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
                
                # Verificar coordenadas válidas
                if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h or x2 <= x1 or y2 <= y1:
                    continue
                    
                face_img = rgb_frame[y1:y2, x1:x2]
                identity = recognize_face(face_img)
                if identity not in identities:
                    identities.append(identity)

        if not identities:
            identities.append("sin rostro")

        emit('response', {'identities': identities})
    except Exception as e:
        print(f"Error en procesamiento de frame: {e}")
        emit('response', {'identities': [f'error: {str(e)}']})

if __name__ == '__main__':
    # Obtener puerto desde variables de entorno (para compatibilidad con servicios de despliegue)
    port = int(os.environ.get('PORT', 5000))
    
    # Configuración para eventlet
    import eventlet
    eventlet.monkey_patch()
    
    # Ejecutar el servidor
    socketio.run(app, host='0.0.0.0', port=port, debug=False)