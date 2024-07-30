import torch
import cv2
import numpy as np
import pathlib
from collections import Counter

# Solución temporal para pathlib en Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Función para encontrar el índice de la cámara
def find_camera_index(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1

try:
    # Encontrar el índice de la cámara externa
    camera_index = find_camera_index()
    if camera_index == -1:
        print("Error: No se pudo encontrar una cámara conectada.")
        exit()

    print(f"Usando la cámara en el índice: {camera_index}")

    # Cargar el modelo
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path="C:/Users/Issac/PycharmProjects/PCar/model/best.pt",
                           force_reload=True)

    # Captura de video
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        exit()

    # Bucle de detección
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir frame (stream end?). Saliendo ...")
            break

        try:
            # Realizar detección
            detection = model(frame)

            # Contabilizar las detecciones
            detection_counter = Counter()
            for det in detection.xyxy[0]:
                class_id = int(det[-1])
                detection_counter[class_id] += 1

            print(f"Detecciones: {detection_counter}")

            # Convertir resultados a numpy array y mostrarlo
            cv2.imshow('Car Detector', np.squeeze(detection.render()))

        except Exception as e:
            print(f"Error en la detección: {e}")

        t = cv2.waitKey(5)
        if t == 27:  # Salir con ESC
            break

except Exception as e:
    print(f"Error al cargar el modelo o iniciar la captura de video: {e}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
