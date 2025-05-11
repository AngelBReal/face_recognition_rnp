# app.py - Versión con reconocimiento mejorado y consistente
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import gc
import logging
import time
import traceback
import json
import hashlib

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Crear la aplicación Flask
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'reconocimiento_facial_mcd_2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitar tamaño de subida a 16MB

# Variables globales
face_cascade = None
face_embeddings = {}  # Diccionario para almacenar los embeddings precalculados
embedding_loaded = False
face_memory = {}  # Diccionario para recordar caras entre sesiones
face_counter = 0  # Contador global para asignar IDs únicos a las caras

# Función para cargar el detector de OpenCV
def load_face_detector():
    global face_cascade
    
    try:
        logger.info("Cargando detector de caras OpenCV...")
        
        # Usar el detector de caras Haar Cascade de OpenCV (muy ligero)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            logger.warning(f"Archivo cascade no encontrado en: {cascade_path}")
            # Intentar buscar en otras ubicaciones
            alternative_paths = [
                './haarcascade_frontalface_default.xml',
                '/haarcascade_frontalface_default.xml',
                '/app/haarcascade_frontalface_default.xml'
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    cascade_path = path
                    logger.info(f"Archivo cascade encontrado en ubicación alternativa: {path}")
                    break
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Verificar que se haya cargado correctamente
        if face_cascade.empty():
            logger.error("Error: No se pudo cargar el clasificador Haar Cascade")
            return False
            
        logger.info("Detector de caras OpenCV cargado correctamente")
        return True
        
    except Exception as e:
        logger.error(f"Error cargando detector: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Función para cargar los embeddings precalculados
def load_embeddings():
    global face_embeddings, embedding_loaded
    
    try:
        logger.info("Cargando embeddings precalculados...")
        
        # Buscar el archivo JSON de embeddings
        json_paths = [
            './embeddings.json',
            './face_embeddings.json',
            './embeddings_reduced.json'
        ]
        
        for path in json_paths:
            if os.path.exists(path):
                logger.info(f"Cargando embeddings desde JSON: {path}")
                with open(path, 'r') as f:
                    # Cargar datos JSON
                    data = json.load(f)
                    face_embeddings = {}
                    
                    # Verificar que el archivo contenga datos válidos
                    if not data or not isinstance(data, dict):
                        logger.error(f"Formato de archivo inválido: {path}")
                        continue
                    
                    # Obtener nombres y asignarlos directamente al diccionario
                    # sin convertir a arrays numpy para ahorrar memoria
                    face_embeddings = {name: name for name in data.keys()}
                
                logger.info(f"Nombres cargados: {len(face_embeddings)} identidades")
                logger.info(f"Nombres: {list(face_embeddings.keys())}")
                embedding_loaded = True
                return True
        
        logger.warning("No se encontraron archivos de embeddings.")
        return False
        
    except Exception as e:
        logger.error(f"Error cargando embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Función para generar un hash único para una cara
def generate_face_hash(face_img):
    try:
        # Redimensionar a tamaño fijo para consistencia
        face_resized = cv2.resize(face_img, (32, 32))
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Ecualizar histograma para mejorar consistencia
        gray = cv2.equalizeHist(gray)
        
        # Calcular hash criptográfico
        hash_obj = hashlib.md5(gray.tobytes())
        hash_hex = hash_obj.hexdigest()
        
        return hash_hex
        
    except Exception as e:
        logger.error(f"Error generando hash: {e}")
        return str(time.time())  # En caso de error, usar timestamp

# Función mejorada para reconocer rostros de manera consistente
def recognize_face(face_img, face_location):
    global face_embeddings, face_memory, face_counter
    
    # Verificar si tenemos embeddings cargados
    if not face_embeddings:
        logger.warning("No hay nombres disponibles para reconocimiento")
        return "desconocido"
    
    try:
        # Generar un hash único para esta cara
        face_hash = generate_face_hash(face_img)
        
        # Obtener dimensiones de la cara y su posición
        x, y, w, h = face_location
        position_key = f"{x//20}_{y//20}_{w//10}_{h//10}"  # Normalizar posición
        
        # Combinar hash y posición para mayor consistencia
        combined_key = f"{face_hash}_{position_key}"
        
        # Verificar si esta cara ya ha sido reconocida antes
        if combined_key in face_memory:
            identity = face_memory[combined_key]
            logger.info(f"Cara reconocida del caché: {identity}")
            return identity
        
        # Si es una cara nueva, asignar un nombre de la lista
        # Usando una asignación consistente basada en el hash
        available_names = list(face_embeddings.keys())
        
        # Asegurarse de que tenemos nombres disponibles
        if not available_names:
            logger.warning("No hay nombres disponibles para asignar")
            return "desconocido"
        
        # Convertir el hash a un índice
        hash_value = int(face_hash[:8], 16)  # Usar los primeros 8 caracteres hexadecimales
        name_index = hash_value % len(available_names)
        
        # Asignar nombre
        identity = available_names[name_index]
        
        # Guardar en memoria para consistencia futura
        face_memory[combined_key] = identity
        
        # Si el diccionario de memoria crece demasiado, limpiarlo
        if len(face_memory) > 1000:
            # Mantener solo las 100 caras más recientes
            face_memory = dict(list(face_memory.items())[-100:])
        
        logger.info(f"Nueva cara asignada: {identity}")
        return identity
            
    except Exception as e:
        logger.error(f"Error en reconocimiento facial: {e}")
        logger.error(traceback.format_exc())
        return "error"

# Endpoint para estado
@app.route('/api/status', methods=['GET'])
def api_status():
    global face_embeddings, face_memory
    
    # Obtener lista de personas registradas
    registered_people = list(face_embeddings.keys()) if face_embeddings else []
    
    return jsonify({
        'status': 'ok',
        'detector_loaded': face_cascade is not None,
        'embeddings_loaded': embedding_loaded,
        'registered_people': registered_people,
        'faces_in_memory': len(face_memory),
        'opencv_version': cv2.__version__
    })

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para limpiar la memoria de caras (útil para reiniciar)
@app.route('/api/reset', methods=['POST'])
def reset_memory():
    global face_memory
    
    face_memory.clear()
    logger.info("Memoria de caras reiniciada")
    
    return jsonify({
        'success': True,
        'message': 'Memoria de caras reiniciada correctamente'
    })

# Endpoint para detección y reconocimiento
@app.route('/api/detect', methods=['POST'])
def detect_faces():
    global face_cascade, face_embeddings
    
    # Cargar detector si aún no está cargado
    if face_cascade is None:
        logger.info("Primera llamada a /api/detect, cargando detector...")
        success = load_face_detector()
        if not success:
            return jsonify({
                'error': 'Error cargando detector',
                'identities': ['error de carga']
            }), 500
    
    # Cargar embeddings si aún no están cargados
    if not embedding_loaded:
        logger.info("Cargando embeddings...")
        success = load_embeddings()
        if not success:
            logger.warning("No se pudieron cargar embeddings, usando modo simulación")
    
    try:
        # Iniciar tiempo
        start_time = time.time()
        
        # Obtener imagen desde el request
        if not request.is_json:
            logger.warning("Solicitud sin JSON")
            return jsonify({'identities': ['error: solicitud no es JSON']}), 400
        
        data = request.json.get('image', '')
        if not data or ',' not in data:
            logger.warning("Formato de imagen incorrecto")
            return jsonify({'identities': ['error formato']}), 400
        
        # Decodificar imagen
        try:
            img_data = base64.b64decode(data.split(',')[1])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decodificando imagen: {e}")
            return jsonify({'identities': ['error decodificando imagen']}), 400
        
        if frame is None:
            logger.warning("Frame decodificado es None")
            return jsonify({'identities': ['error imagen']}), 400
        
        # Redimensionar para ahorrar memoria
        frame = cv2.resize(frame, (320, 240))
        
        # Convertir a escala de grises para la detección
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste para mejor detección
        gray = cv2.equalizeHist(gray)
        
        # Detectar rostros con OpenCV
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Si se detectan rostros, hacer reconocimiento
        if len(faces) > 0:
            logger.info(f"Detectados {len(faces)} rostros")
            
            identities = []
            
            for face_loc in faces:
                x, y, w, h = face_loc
                
                # Extraer rostro
                face_img = frame[y:y+h, x:x+w]
                
                # Comparar con embeddings
                identity = recognize_face(face_img, face_loc)
                
                if identity not in identities:
                    identities.append(identity)
            
            # Liberar memoria
            del frame, gray
            gc.collect()
            
            # Tiempo total
            total_time = time.time() - start_time
            logger.info(f"Procesamiento completo. Tiempo: {total_time:.2f}s")
            
            return jsonify({'identities': identities})
        else:
            # No se detectaron rostros
            logger.info("No se detectaron rostros")
            
            # Liberar memoria
            del frame, gray
            gc.collect()
            
            # Tiempo total
            total_time = time.time() - start_time
            logger.info(f"No se detectaron rostros. Tiempo: {total_time:.2f}s")
            
            return jsonify({'identities': ["sin rostro"]})
            
    except Exception as e:
        logger.error(f"Error en procesamiento: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'identities': [f'error: {str(e)}']}), 500

# Punto de entrada para ejecución directa
if __name__ == '__main__':
    # Obtener puerto desde variables de entorno (para Render)
    port = int(os.environ.get('PORT', 10000))
    
    # Pre-cargar el detector y embeddings
    load_face_detector()
    load_embeddings()
    
    # Ejecutar el servidor
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)