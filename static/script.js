const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const output = document.getElementById('output');
const switchBtn = document.getElementById('switchCamera');
const context = canvas.getContext('2d');
const socket = io();

let facingMode = 'user'; // o 'environment'

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { facingMode } })
        .then(stream => { video.srcObject = stream; })
        .catch(err => console.error('Error:', err));
}

switchBtn.addEventListener('click', () => {
    facingMode = (facingMode === 'user') ? 'environment' : 'user';
    startCamera();
});

startCamera();

video.addEventListener('play', () => {
    setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');
        socket.emit('frame', dataURL);
    }, 800);
});

socket.on('response', data => {
    const names = data.identities.join(', ');
    output.innerText = `Detectados: ${names}`;
});
