import gymnasium as gym
import time
import mss
import numpy as np
import cv2
import pyautogui

Y_MIN_INSPECCION = 35 + 22  # Nueva Altura mínima
Y_MAX_INSPECCION = 110 + 22  # Nueva Altura máxima
#Punto de referencia del dinosaurio
X_DINO = 80
LIMITE_VISION_X = 550
ROI_DINO_GAME = {"top": 180, "left": 600, "width": 600, "height": 190}


class DinoEntorno(gym.Env):
    action_space = gym.spaces.Discrete(2)  # 0: Nada, 1: Saltar

    # [X_inicio, X_fin, Y_max, Y_min]
    observation_space = gym.spaces.Box(low=0, high=600, shape=(4,), dtype=np.int32)

    def __init__(self, template_path="game_over_template.png"):
        super(DinoEntorno, self).__init__()
        self.sct = mss.mss()
        self.roi = ROI_DINO_GAME

        # Cargar la plantilla GAME OVER
        try:
            self.game_over_template = cv2.imread(template_path, 0)
            if self.game_over_template is None:
                print(
                    f"ERROR: No se pudo cargar la plantilla '{template_path}'. ¿Existe?")
        except Exception as e:
            print(f"Error al cargar la plantilla: {e}")
            self.game_over_template = None

    def _get_screenshot(self):
        sct_img = self.sct.grab(self.roi)
        img = np.array(sct_img)#Se convierte la imagen en un array
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #0 y 255

        # Binarización: 0=obstáculo, 255=fondo
        _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)

        return binary_img, gray_img

    def _get_observation(self):
        binary_img, _ = self._get_screenshot()

        # Inicializar variables de seguimiento
        min_x_obstaculo = LIMITE_VISION_X
        max_x_obstaculo = X_DINO
        min_y_obstaculo = Y_MAX_INSPECCION
        max_y_obstaculo = Y_MIN_INSPECCION

        obstaculo_encontrado = False

        # Recorrer el área de visión completa
        for x in range(X_DINO, LIMITE_VISION_X):
            # Buscamos obstáculos en el rango
            obstaculo_en_columna = np.where(binary_img[Y_MIN_INSPECCION:Y_MAX_INSPECCION, x] == 0)[0]

            if len(obstaculo_en_columna) > 0:
                obstaculo_encontrado = True

                # Actualizar límites (horizontal)
                if x < min_x_obstaculo:
                    min_x_obstaculo = x
                if x > max_x_obstaculo:
                    max_x_obstaculo = x

                # Actualizar límites (vertical)
                y_max_global = Y_MIN_INSPECCION + np.max(obstaculo_en_columna)
                y_min_global = Y_MIN_INSPECCION + np.min(obstaculo_en_columna)

                if y_max_global > max_y_obstaculo:
                    max_y_obstaculo = y_max_global
                if y_min_global < min_y_obstaculo:
                    min_y_obstaculo = y_min_global

        # Construir la observación
        if obstaculo_encontrado:
            distancia_inicio = min_x_obstaculo - X_DINO
            distancia_fin = max_x_obstaculo - X_DINO

            # Devuelve [X_inicio, X_fin, Y_max, Y_min]
            return np.array([distancia_inicio, distancia_fin, max_y_obstaculo, min_y_obstaculo], dtype=np.int32)

        # Caso: Camino Despejado
        max_distancia = LIMITE_VISION_X - X_DINO
        altura_segura = Y_MAX_INSPECCION

        # Devuelve valores seguros cuando no hay obstáculo
        return np.array([max_distancia, max_distancia, altura_segura, altura_segura], dtype=np.int32)

    def check_game_over(self):
        """Detecta GAME OVER  con Template Matching."""
        if self.game_over_template is None:
            return False

        _, gray_img = self._get_screenshot()

        res = cv2.matchTemplate(gray_img, self.game_over_template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0:
            return True

        return False

    def step(self, action):
        if action == 1:
            pyautogui.press('space')

        time.sleep(0.01)
        observation = self._get_observation()
        reward = 1
        done = self.check_game_over()

        if done:
            reward = -500

        info = {}
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pyautogui.press('space')
        observation = self._get_observation()
        info = {}
        return observation, info

    def close(self):
        pass