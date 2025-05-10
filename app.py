# app.py - Versión REST optimizada para bajo consumo de memoria
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import pickle
import os
import gc

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'reconocimiento_facial_mcd_2025'

# Variables globales para los modelos
embedder = None
database = None
face_detection = None

def load_models():
    global embedder, database, face_detection
    
    # Cargar modelos solo cuando sea necesario
    try:
        print("Cargando modelos...")
        
        # Cargar MediaPipe primero porque es más ligero
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Cargar FaceNet (esto consume más memoria)
        from keras_facenet import FaceNet
        embedder = FaceNet()
        
        # Cargar base de datos de caras
        with open('face_database.pkl', 'rb') as f:
            database = pickle.load(f)
            
        print("Modelos cargados correctamente")
        
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        raise

def recognize_face(face_img):
    global embedder, database
    
    # Verificar que tenemos una imagen válida
    if face_img is None or face_img.size == 0:
        return 'desconocido'
    
    try:
        # Redimensionar para FaceNet
        face_img = cv2.resize(face_img, (160, 160))
        
        # Obtener embedding
        embedding = embedder.embeddings([face_img])[0]
        
        # Buscar la cara más cercana en la base de datos
        min_dist = 100
        identity = 'desconocido'
        
        for name, db_embeddings in database.items():
            for db_emb in db_embeddings:
                dist = np.linalg.norm(embedding - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    identity = name
        
        # Umbral de similitud
        if min_dist > 0.7:
            identity = 'desconocido'
        
        # Liberar memoria
        gc.collect()
        
        return identity
    except Exception as e:
        print(f"Error en reconocimiento facial: {e}")
        return 'error'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    global face_detection
    
    # Cargar modelos si aún no están cargados
    if face_detection is None:
        try:
            load_models()
        except Exception as e:
            return jsonify({'identities': [f'Error cargando modelos: {str(e)}']}), 500
    
    try:
        # Obtener imagen desde el request
        data = request.json.get('image', '')
        
        if not data or ',' not in data:
            return jsonify({'identities': ['error formato']}), 400
            
        # Decodificar imagen
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'identities': ['error imagen']}), 400
        
        # Redimensionar para ahorrar memoria
        frame = cv2.resize(frame, (160, 120))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros
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
                    
                # Extraer rostro y reconocer
                face_img = rgb_frame[y1:y2, x1:x2]
                identity = recognize_face(face_img)
                if identity not in identities:
                    identities.append(identity)

        if not identities:
            identities.append("sin rostro")

        # Liberar memoria explícitamente
        del frame, rgb_frame, results
        gc.collect()

        return jsonify({'identities': identities})
    except Exception as e:
        print(f"Error en procesamiento de frame: {e}")
        return jsonify({'identities': [f'error: {str(e)}']}), 500

if __name__ == '__main__':
    # Obtener puerto desde variables de entorno (para Render)
    port = int(os.environ.get('PORT', 10000))
    
    # Ejecutar el servidor en modo producción
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)