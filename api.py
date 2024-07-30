import torch
import cv2
import numpy as np
import pathlib
from collections import Counter
import mss
import time
import tkinter as tk
from tkinter import messagebox

# Solución temporal para pathlib en Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Función para encontrar los índices de las cámaras
def find_camera_indices(max_index=10):
    indices = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            cap.release()
    return indices

# Función para capturar pantalla
def capture_screen(region=None):
    with mss.mss() as sct:
        screenshot = sct.grab(region) if region else sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

# Función principal de detección
def detect(mode):
    try:
        # Encontrar los índices de las cámaras conectadas
        camera_indices = find_camera_indices()
        if not camera_indices:
            print("Error: No se pudo encontrar ninguna cámara conectada.")
            exit()

        print(f"Cámaras encontradas en los índices: {camera_indices}")

        # Cargar el modelo
        model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:/Users/Issac/PycharmProjects/PCar/model/best.pt", force_reload=True)

        # Inicializar contadores de detecciones
        detection_counter = Counter()

        # Configurar la captura de video
        caps = [cv2.VideoCapture(idx) for idx in camera_indices]
        screen_region = {'top': 100, 'left': 100, 'width': 1280, 'height': 760}  # Ajustar según la ubicación de la videollamada

        # Configurar la ventana de visualización
        cv2.namedWindow('Detección', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detección', 1280, 760)

        # Bucle de detección
        while True:
            frames = []

            if mode == 'camera':
                # Leer frames de las cámaras
                for cap in caps:
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        frames.append(None)
            elif mode == 'videocall':
                # Leer frame de la pantalla (simulando videollamada de Telegram)
                screen_frame = capture_screen(screen_region)
                frames.append(screen_frame)

            start_time = time.time()

            for frame in frames:
                if frame is not None:
                    try:
                        # Realizar detección
                        detection = model(frame)

                        # Contabilizar las detecciones
                        for det in detection.xyxy[0]:  # iterar sobre detecciones
                            class_id = int(det[-1])
                            detection_counter[class_id] += 1

                        print(f"Detecciones: {detection_counter}")

                        # Convertir resultados a numpy array y mostrarlo
                        cv2.imshow('Detección', np.squeeze(detection.render()))

                    except Exception as e:
                        print(f"Error en la detección: {e}")

            t = cv2.waitKey(1)
            if t == 27:  # Salir con ESC
                break

            # Mantener una tasa de refresco de más de 30 fps
            elapsed_time = time.time() - start_time
            if elapsed_time < 1/30:
                time.sleep(1/30 - elapsed_time)

        print("Contador de detecciones:", detection_counter)

    except Exception as e:
        print(f"Error al cargar el modelo o iniciar la captura de video: {e}")

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

# Función para iniciar la detección con cámaras
def start_camera_detection():
    detect('camera')

# Función para iniciar la detección con videollamada
def start_videocall_detection():
    detect('videocall')

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Detección de Objetos con YOLOv5")

description = """
Este modelo utiliza YOLOv5 para detectar objetos en tiempo real.
Puedes elegir entre dos modos de detección:
1. Detección usando las cámaras conectadas.
2. Detección simulada de una videollamada de Telegram.
"""

label = tk.Label(root, text=description, justify=tk.LEFT)
label.pack(pady=10)

button_camera = tk.Button(root, text="Detectar con Cámaras", command=start_camera_detection)
button_camera.pack(pady=5)

button_videocall = tk.Button(root, text="Detectar con Videollamada", command=start_videocall_detection)
button_videocall.pack(pady=5)

root.mainloop()
