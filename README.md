
### 📄 README.md

# 🎥 Proyecto de Reconocimiento Facial - MCD

Este proyecto implementa un sistema de reconocimiento facial en tiempo real usando Flask, Mediapipe y FaceNet.  
Funciona en computadoras y teléfonos móviles, permite cambiar entre cámaras y muestra los nombres de las personas detectadas.

---

### 🚀 Funcionalidades principales

✅ Captura de video en tiempo real (WebRTC)  
✅ Detección de múltiples rostros simultáneamente  
✅ Identificación de personas registradas en la base  
✅ Muestra de nombres detectados en la interfaz  
✅ Optimización para correr en Render o localmente  
✅ Diseño web con logo, footer y links a GitHub / LinkedIn / Escuela

---

### 💻 Tecnologías usadas

- Python
- Flask + Flask-SocketIO
- Mediapipe
- Keras-FaceNet
- HTML5 + CSS + JavaScript

---

### 📓 Notebook de embeddings

📄 [Ver el notebook en Google Colab](https://colab.research.google.com/drive/15EgZHAOr41qktQks3yr85aWkLnKJfq3Z#scrollTo=dD1rYHCZ9p7D)

---

### 📦 Cómo correr localmente

#### 1️⃣ Clona el proyecto

```bash
git clone https://github.com/TuUsuario/TuRepo.gi¨
```

#### 2️⃣ Crea entorno virtual
``` bash
python -m venv venv
source venv/bin/activate  
```

### En Windows: venv\Scripts\activate

#### 3️⃣ Instala dependencias
``` bash
pip install -r requirements.txt
```

#### 4️⃣ Corre la app
``` bash
python app.py
```

#### 5️⃣ Abre en navegador
http://localhost:5000

---

### 📂 Archivos incluidos

✅ **Código principal**

* `app.py`: Código completo de la aplicación, con comentarios explicativos de cada sección.
* `embeddings.pkl`: Archivo generado con el notebook con embeddings para el reconocimiento.
* `requirements.txt`: Lista de dependencias necesarias.
* `notebook_embeddings.ipynb`: Colab Notebook usado para crear los embeddings.

✅ **Archivo HTML**

* `templates/index.html`: Archivo que sirve como interfaz web para la aplicación.
  Permite capturar video en vivo, detectar rostros, mostrar nombres detectados y cambiar de cámara.

---


---

### 🌐 Demo en línea

🚀 [Ver app desplegada en Render](https://tusitio.render.com)

---

### 🙌 Créditos

* Universidad: [Mi Escuela](https://mcd.unison.mx/)
* Autor: [Tu Nombre](https://www.linkedin.com/in/angelbarrazareal/)
* GitHub: [@TuUsuario](https://github.com/AngelBReal)

---

### 📃 Licencia

Este proyecto es de uso académico. Puedes adaptarlo, mejorar su diseño y usarlo como base para tus propios proyectos.





