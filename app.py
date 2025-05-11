from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import gc
import logging
import time
import traceback
import psutil

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Crear la aplicación Flask
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'reconocimiento_facial_mcd_2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitar tamaño de subida a 16MB

# Variables globales
face_detection = None

# Estado interno para simulación
SIMULATED_FACES = {
    "USUARIO1": "Juan Pérez",
    "USUARIO2": "María García",
    "USUARIO3": "Carlos Rodríguez"
}
current_user_index = 0

# Endpoint para diagnóstico y estado
@app.route('/api/status', methods=['GET'])
def api_status():
    # Verificar memoria disponible
    memory = psutil.virtual_memory()
    
    # Verificar si los modelos están cargados
    return jsonify({
        'status': 'ok',
        'memory': {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent
        },
        'models_loaded': face_detection is not None,
        'mediapipe_only': True,  # Indicar que estamos en modo ligero
        'opencv_version': cv2.__version__
    })

# Función para cargar solo el detector de MediaPipe (ligero)
def load_face_detector():
    global face_detection
    
    try:
        logger.info("Iniciando carga de MediaPipe Face Detection...")
        start_time = time.time()
        
        # Cargar MediaPipe - mucho más ligero que FaceNet
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=0  # 0=modelo ligero para caras cercanas
        )
        
        # Calcular tiempo total
        elapsed_time = time.time() - start_time
        logger.info(f"MediaPipe cargado. Tiempo: {elapsed_time:.2f} segundos")
        
        # Forzar recolección de basura para liberar memoria
        gc.collect()
        
        return True
    except Exception as e:
        logger.error(f"Error cargando detector: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Función para simular reconocimiento
def simulate_recognition():
    global current_user_index
    
    # Simular diferentes usuarios en secuencia
    users = list(SIMULATED_FACES.values())
    result = users[current_user_index % len(users)]
    
    # Avanzar al siguiente usuario para la próxima llamada
    current_user_index += 1
    
    return result

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint simple sin ML
@app.route('/api/detect-mock', methods=['POST'])
def detect_faces_mock():
    try:
        # Simulamos detección
        identity = simulate_recognition()
        return jsonify({'identities': [identity]})
    except Exception as e:
        logger.error(f"Error en endpoint mock: {e}")
        return jsonify({'identities': [f'error: {str(e)}']}), 500

# Endpoint para detección (sin reconocimiento)
@app.route('/api/detect', methods=['POST'])
def detect_faces():
    global face_detection
    
    # Cargar detector si aún no está cargado
    if face_detection is None:
        logger.info("Primera llamada a /api/detect, cargando detector...")
        success = load_face_detector()
        if not success:
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros con MediaPipe
        results = face_detection.process(rgb_frame)

        # Si se detectan rostros, simulamos reconocimiento
        if results.detections and len(results.detections) > 0:
            logger.info(f"Detectados {len(results.detections)} rostros")
            
            # Simulamos reconocimiento (para demostración)
            identity = simulate_recognition()
            
            # Liberar memoria
            del frame, rgb_frame, results
            gc.collect()
            
            return jsonify({'identities': [identity]})
        else:
            # No se detectaron rostros
            logger.info("No se detectaron rostros")
            
            # Liberar memoria
            del frame, rgb_frame, results
            gc.collect()
            
            return jsonify({'identities': ["sin rostro"]})
            
    except Exception as e:
        logger.error(f"Error en procesamiento: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'identities': [f'error: {str(e)}']}), 500

# Punto de entrada para ejecución directa
if __name__ == '__main__':
    # Obtener puerto desde variables de entorno (para Render)
    port = int(os.environ.get('PORT', 10000))
    
    # Pre-cargar el detector (opcional)
    load_face_detector()
    
    # Ejecutar el servidor
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)