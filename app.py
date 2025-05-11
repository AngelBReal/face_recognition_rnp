# app.py - Versión minimalista con solo OpenCV
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import gc
import logging
import time
import traceback

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

# Nombres simulados para la demostración
DEMO_NAMES = ["Juan Pérez", "María García", "Carlos Rodríguez", "Ana Martínez"]
current_name_index = 0

# Función para cargar el detector de OpenCV
def load_face_detector():
    global face_cascade
    
    try:
        logger.info("Cargando detector de caras OpenCV...")
        
        # Usar el detector de caras Haar Cascade de OpenCV (muy ligero)
        # Este archivo debería estar en tu carpeta del proyecto
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            logger.error(f"Archivo cascade no encontrado en: {cascade_path}")
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

# Simulación simple de reconocimiento para la demostración
def simulate_recognition():
    global current_name_index
    
    # Rotar entre los nombres simulados
    name = DEMO_NAMES[current_name_index % len(DEMO_NAMES)]
    current_name_index += 1
    
    return name

# Endpoint para estado
@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'ok',
        'detector_loaded': face_cascade is not None,
        'detector_type': 'OpenCV Haar Cascade (minimal)',
        'opencv_version': cv2.__version__
    })

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para detección simulada (sin ML)
@app.route('/api/detect-mock', methods=['POST'])
def detect_faces_mock():
    try:
        # Simulamos detección sin procesar la imagen
        identity = simulate_recognition()
        return jsonify({'identities': [identity]})
    except Exception as e:
        logger.error(f"Error en endpoint mock: {e}")
        return jsonify({'identities': [f'error: {str(e)}']}), 500

# Endpoint para detección básica con OpenCV
@app.route('/api/detect', methods=['POST'])
def detect_faces():
    global face_cascade
    
    # Cargar detector si aún no está cargado
    if face_cascade is None:
        logger.info("Primera llamada a /api/detect, cargando detector...")
        success = load_face_detector()
        if not success:
            logger.error("No se pudo cargar el detector de caras")
            return jsonify({
                'error': 'Error cargando detector',
                'identities': ['error de carga']
            }), 500
    
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
        frame = cv2.resize(frame, (160, 120))
        
        # Convertir a escala de grises para la detección
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros con OpenCV
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Si se detectan rostros, simulamos reconocimiento
        if len(faces) > 0:
            logger.info(f"Detectados {len(faces)} rostros")
            
            # Simulamos reconocimiento (para demostración)
            identity = simulate_recognition()
            
            # Liberar memoria
            del frame, gray
            gc.collect()
            
            # Tiempo total
            total_time = time.time() - start_time
            logger.info(f"Procesamiento completo. Tiempo: {total_time:.2f}s")
            
            return jsonify({'identities': [identity]})
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
    
    # Pre-cargar el detector
    load_face_detector()
    
    # Ejecutar el servidor
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)