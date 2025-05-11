#!/bin/bash

# Script para preparar la aplicación en entorno de Render

echo "Iniciando script de construcción..."

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar script para descargar el clasificador Haar
python download_cascade.py

# Verificar que el archivo existe
if [ -f "haarcascade_frontalface_default.xml" ]; then
    echo "Verificado: Archivo Haar Cascade listo para usar."
else
    # Intentar descarga directa como último recurso
    echo "Descargando directamente del repositorio de OpenCV..."
    curl -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
    
    if [ -f "haarcascade_frontalface_default.xml" ]; then
        echo "Descarga directa exitosa."
    else
        echo "ADVERTENCIA: No se pudo obtener el archivo para detección facial."
    fi
fi

echo "Construcción completada."