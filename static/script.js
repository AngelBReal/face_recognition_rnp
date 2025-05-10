const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const output = document.getElementById('output');
const switchBtn = document.getElementById('switchCamera');
const context = canvas.getContext('2d');

let facingMode = 'user'; // o 'environment'
let isProcessing = false;
let intervalId = null;
let serverAvailable = true;
let retryCount = 0;
const MAX_RETRIES = 3;

// Iniciar cámara
function startCamera() {
    // Detener el stream actual si existe
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    
    // Solicitar acceso a la cámara
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: facingMode,
            width: { ideal: 320 },  // Resolución reducida para ahorrar ancho de banda
            height: { ideal: 240 }
        }
    })
    .then(stream => {
        video.srcObject = stream;
        video.play();
        
        // Una vez que la cámara está lista, empezamos a procesar
        video.onloadeddata = () => {
            startProcessing();
            output.innerText = 'Cámara activada, esperando detección...';
            output.style.color = '#00ff00';
        };
    })
    .catch(err => {
        console.error('Error al acceder a la cámara:', err);
        output.innerText = 'Error: No se pudo acceder a la cámara';
        output.style.color = '#ff0000';
    });
}

// Botón para cambiar de cámara
switchBtn.addEventListener('click', () => {
    facingMode = (facingMode === 'user') ? 'environment' : 'user';
    output.innerText = 'Cambiando cámara...';
    startCamera();
});

// Iniciar procesamiento
function startProcessing() {
    if (intervalId) {
        clearInterval(intervalId);
    }
    
    // Intervalo más largo para reducir la carga del servidor
    intervalId = setInterval(() => {
        if (!isProcessing && video.readyState === 4 && serverAvailable) {
            processFrame();
        }
    }, 1000);
}

// Detener procesamiento
function stopProcessing() {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
}

// Procesar un frame y enviarlo al servidor
async function processFrame() {
    try {
        isProcessing = true;
        
        // Dibujar el frame en el canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convertir a formato JPEG con calidad reducida para menor tamaño
        const dataURL = canvas.toDataURL('image/jpeg', 0.5);
        
        // Mostrar indicador de procesamiento
        output.innerText = 'Procesando...';
        
        // Enviar al servidor usando fetch
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataURL }),
        });
        
        if (!response.ok) {
            throw new Error(`Error HTTP: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Mostrar resultados
        const names = data.identities.join(', ');
        output.innerText = `Detectados: ${names}`;
        output.style.color = '#00ff00';
        
        // Reiniciar contador de reintentos
        retryCount = 0;
        
        // Marcar servidor como disponible
        serverAvailable = true;
    } catch (err) {
        console.error('Error en procesamiento:', err);
        
        retryCount++;
        if (retryCount >= MAX_RETRIES) {
            output.innerText = 'Error de conexión con el servidor';
            output.style.color = '#ff0000';
            
            // Marcar servidor como no disponible temporalmente
            serverAvailable = false;
            
            // Intentar reconectar después de un tiempo
            setTimeout(() => {
                serverAvailable = true;
                output.innerText = 'Reconectando...';
                retryCount = 0;
            }, 5000);
        } else {
            output.innerText = `Reintentando (${retryCount}/${MAX_RETRIES})...`;
            output.style.color = '#ffaa00';
        }
    } finally {
        // Permitir que se procese el siguiente frame
        isProcessing = false;
    }
}

// Función para verificar la disponibilidad del servidor
async function checkServer() {
    try {
        const response = await fetch('/', { 
            method: 'GET',
            // Añadir una consulta aleatoria para evitar la caché
            cache: 'no-store' 
        });
        
        if (response.ok) {
            serverAvailable = true;
            if (output.innerText.includes('Error de conexión')) {
                output.innerText = 'Conectado: esperando detección...';
                output.style.color = '#00ff00';
            }
        }
    } catch (err) {
        serverAvailable = false;
        output.innerText = 'Error de conexión con el servidor';
        output.style.color = '#ff0000';
    }
}

// Verificar servidor cada 15 segundos
setInterval(checkServer, 15000);

// Iniciar cámara al cargar
window.addEventListener('load', () => {
    output.innerText = 'Iniciando cámara...';
    startCamera();
    
    // Verificar servidor inicial
    checkServer();
});