const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const output = document.getElementById('output');
const switchBtn = document.getElementById('switchCamera');
const context = canvas.getContext('2d');

// Configuración del socket
const socket = io({
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    timeout: 20000
});

let facingMode = 'user'; // o 'environment'
let isProcessing = false;
let intervalId = null;

// Estado de conexión
socket.on('connect', () => {
    console.log('Conectado al servidor');
    output.innerText = 'Conectado: esperando detección...';
    output.style.color = '#00ff00';
    startSendingFrames();
});

socket.on('disconnect', () => {
    console.log('Desconectado del servidor');
    output.innerText = 'Desconectado: intentando reconectar...';
    output.style.color = '#ff0000';
    stopSendingFrames();
});

socket.on('connect_error', (err) => {
    console.error('Error de conexión:', err);
    output.innerText = 'Error de conexión: intentando reconectar...';
    output.style.color = '#ff0000';
});

// Iniciar cámara
function startCamera() {
    // Detener el stream actual si existe
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: facingMode,
            width: { ideal: 640 },
            height: { ideal: 480 }
        }
    })
    .then(stream => {
        video.srcObject = stream;
        video.play();
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
    startCamera();
});

// Iniciar envío de frames
function startSendingFrames() {
    if (intervalId) {
        clearInterval(intervalId);
    }
    
    intervalId = setInterval(() => {
        if (!isProcessing && video.readyState === 4) {
            sendFrame();
        }
    }, 800);
}

// Detener envío de frames
function stopSendingFrames() {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
}

// Enviar un frame al servidor
function sendFrame() {
    try {
        isProcessing = true;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const dataURL = canvas.toDataURL('image/jpeg', 0.7);
        socket.emit('frame', dataURL);
    } catch (err) {
        console.error('Error al procesar frame:', err);
        isProcessing = false;
    }
}

// Recibir respuesta del servidor
socket.on('response', data => {
    const names = data.identities.join(', ');
    output.innerText = `Detectados: ${names}`;
    isProcessing = false;
});

// Iniciar cámara al cargar
startCamera();