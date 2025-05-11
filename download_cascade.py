# Script para descargar el archivo Haar Cascade de OpenCV
import urllib.request
import os
import cv2
import shutil

print("Descargando archivo Haar Cascade para detección facial...")

# Origen del archivo (en la instalación de OpenCV)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

if os.path.exists(cascade_path):
    # Copiar a la carpeta actual
    print(f"Copiando desde: {cascade_path}")
    shutil.copy(cascade_path, './haarcascade_frontalface_default.xml')
    print("Archivo copiado correctamente.")
else:
    # Si no está disponible, descargarlo desde GitHub
    print(f"Archivo no encontrado en: {cascade_path}")
    print("Descargando desde GitHub...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')
    print("Archivo descargado correctamente.")

# Verificar que el archivo existe
if os.path.exists('./haarcascade_frontalface_default.xml'):
    print("Verificación: Archivo disponible en la carpeta del proyecto.")
    print(f"Tamaño: {os.path.getsize('./haarcascade_frontalface_default.xml')} bytes")
else:
    print("Error: No se pudo crear el archivo.")