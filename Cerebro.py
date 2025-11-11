import time
import mss
import numpy as np
import cv2
import pyautogui
from tensorflow.python.data.experimental.ops.testing import sleep

X_DINO = 50 #Pixel donde esta el dinosaurio
DISTANCIA_SALTO =95    #Distancia en pixeles delante del dinosaurio
ROI_DINO_GAME = {"top": 180, "left": 600, "width": 600, "height": 190}
Collision_Y=120  #Linea del suelo
deteccionFondo = 100

def detectar_obstaculo_y_saltar(binary_img):
    #RECORRE LA LINEA DEL SUELO

    for x in range(X_DINO, X_DINO + DISTANCIA_SALTO):
        #BINARY IMG DEVUELVE EL VALOR DEL PIXEL (255 SI ES SUELO, 0 SI ES OBSTACULO)
        if binary_img[Collision_Y, x] == 0:            #Al detectar un obstaculo salta
            pyautogui.press('space')
            return



def procesar_frame():
    time.sleep(0.001)
    with mss.mss() as sct:
        while True:
            sct_img = sct.grab(ROI_DINO_GAME) #Captura el frame en tiempo real
            img = np.array(sct_img) #se convierte en una estructura de datos para el procesamiento con OpenCV (como lo que hicimos con excel)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #lo convierte en escala de grises.
            # convierte todo en binarios (blanco y negro)

            _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV  )
            if binary_img[deteccionFondo, 140] == 0:
                _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

            detectar_obstaculo_y_saltar(binary_img)
            cv2.imshow("Obstaculo y saltar", binary_img)
            #Detener el bucle si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
procesar_frame()