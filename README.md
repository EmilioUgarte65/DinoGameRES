<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/4c3824e3-0beb-4441-9edc-b67da831a9cb"/>

# DinoGameRES
Código para la resolución automática del juego del Dinosaurio, resolución con IA entrenada y resolución con Código.

# Requerimientos.
### versión de Python.

Python versión 3.9

### librerías necesarias.

* Stable-Baseline3 2.7.0: Implementa algoritmos de aprendizaje por refuerzo en PyTorch  
* PyTorch 2.8.0+cpu: Es la versión del framework PyTorch optimizada para ejecutarse únicamente en CPU.  
* Gymnasium 1.1.1: Provee un conjunto estándar de entornos y una API para desarrollar y entrenar algoritmos de aprendizaje por refuerzo.   
* opencv-python 4.12.0.88 : Es una librería de visión por computadora y procesamiento de imágenes.   
* mss 10.1.0: Permite capturar la pantalla completa o regiones específicas.  
* pynput 1.8.1: Simular pulsaciones de teclas.    
* pyautogui 0.9.54: Controlar personajes o realizar acciones automáticas.    
* numpy 2.0.2.
  
El agente fue entrenado solamente con la CPU (13th Gen Intel(R) Core(TM) i7-13700HX).  
Resolución de pantalla 1920 * 1200 (Esto es importante ya que usamos los pixeles en pantalla en puntos específicos para que el Código funcione). En caso de 
# Instrucciones de Uso

### Cerebro

Este script se usara para verificar que el programa detecte el rectángulo del juego y los obstáculos ya que este mismo muestra la imagen mientas juega para poder ver los pixeles que la IA tomara de referencia, de igual manera esta es la forma en la que el juego se hace de manera "Manual".  

<img width="750" height="140" alt="image" src="https://github.com/user-attachments/assets/78bae4c5-b716-4b03-acd9-e04b098171d8" />  

ROI_DINO_GAME Tiene los valores correspondientes al rectángulo que vera nuestro programa.  
X_DINO Tiene la ubicación del dinosaurio.  
DISTANCIA_SALTO =95    Distancia en pixeles delante del dinosaurio (Este se puede ser cambiado de así requerirlo si el dinosaurio no salta bien el obstáculo y se queda a medias).  
Collision_Y es la línea donde se detectarán los obstáculos.

<img width="770" height="656" alt="image" src="https://github.com/user-attachments/assets/76d13308-1ab8-43b6-b967-a890955bd90d" />


Nota: En caso de tener la misma resolución de pantalla se recomienda usar Google Chrome sin la barra de favoritos.

# Parámetros del modelo
Algoritmo: DQN (Deep QNetwork).  
### Parámetros clave de Entrenamiento.py:  
* Tasa de Aprendizaje (learning_rate): 0.0005.  
* Tamaño del Buffer de Replay (buffer_size): 100,000. 
* Exploración final (exploration_final_eps): 0.01.  
* Timesteps Totales: 800,000.

# Estructura de observación.

El agente aprende de un espacio de observación con forma 4 de tipo int 32.  
X_inicio: Distancia en pixeles al borde inicial más cercano al dinosaurio (el inicio del obstáculo).  
X_fin: Distancia en pixeles al borde final más lejano del obstáculo.  
Y_max: Altura máxima del obstáculo (Borde inferior más cercano al suelo).  
Y_min: Altura mínima del obstáculo (Borde superior del obstáculo).  
En caso de no haber obstáculos devuelve valores "seguros".  

Para pasarle información al agente sobre el movimiento y la velocidad de los objetos, el script de entrenamiento (Entrenamiento.py) y el de juego (JuegaIA.py) utiliza la herramienta VecFrameStack con n_stack = 4.  
* El agente recibe las 4 ultimas observaciones base de manera secuencial.
* La red neuronal recibe un vector de 16 valores (Los 4 valores * los 4 frames que se le da).



#   DinoEntorno 
### Class DinoEntrono  
Línea 17: 

    action_space = gym.spaces.Discrete(2)  # 0: Nada, 1: Saltar

Esta línea de código define las acciones que la IA puede realizar en el entorno del juego.  

    observation_space = gym.spaces.Box(low=0, high=600, shape=(4,), dtype=np.int32)

Después de esto se le pasa a la IA el espacio de observación del entorno representando la información en un vector continuo dentro de un rango definido en low = 0 y High = 600 siendo un rango de 0 a 600. El shape especifica que la observación es un vector con 4 elementos (Representando las métricas del obstáculo, Alto y ancho, tomado el inicio y el final de estos dos)  

# Init
Líneas : 22 - 35
  
      def __init__(self, template_path="game_over_template.png"):
              super(DinoEntorno, self).__init__()
              self.sct = mss.mss()
              self.roi = ROI_DINO_GAME


Aquí inicializaremos el entorno para que pueda interactuar con el agente(IA) de reforzamiento de aprendizaje.   

        # Cargar la plantilla GAME OVER
        try:
            self.game_over_template = cv2.imread(template_path, 0)
            if self.game_over_template is None:
                print(
                    f"ERROR: No se pudo cargar la plantilla '{template_path}'. ¿Existe?")
        except Exception as e:
            print(f"Error al cargar la plantilla: {e}")
            self.game_over_template = None  
A su vez dentro del init con un try cargamos la plantilla del game over

# _get_screenshot 
Líneas: 37 - 45

            sct_img = self.sct.grab(self.roi)
            img = np.array(sct_img)#Se convierte la imagen en un array
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #0 y 255
            # Binarización: 0=obstáculo, 255=fondo
            _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
            return binary_img, gray_img

Se toma una screenshot del rectángulo establecido por roi, esa screenshot se convierte en un array y después se convierte en un array en escala de grises.  
Al final se convierte en blanco y negro la imagen y se invierten los colores ya que el fondo del navegador es negro (Caso contrario en lugar de usar  _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV) ) Usar:

     _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)


# _get_observation
Líneas: 47 -94.  
La función principal es detectar el obstáculo más cercano al dinosaurio y devolver sus coordenadas.
### for de recorrido de área de visión. línea 59

        for x in range(X_DINO, LIMITE_VISION_X):
            # Buscamos obstáculos en el rango
            obstaculo_en_columna = np.where(binary_img[Y_MIN_INSPECCION:Y_MAX_INSPECCION, x]              == 0)[0]

            if len(obstaculo_en_columna) > 0:
                obstaculo_encontrado = True
Recorre el área de visión completa y busca obstáculos dentro del rango (en caso de encontrar un obstáculo pone la variable obstaculo_encontrado como true.   

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
Actualiza limites verticales y horizontales.  

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

Si encontró un obstáculo devuelve el ancho del obstáculo en x y el largo del obstáculo en y, en un array tipo int 32.  
En caso de tener el camino despejado, devuelve los valores "seguros", igual en un array tipo int 32.

# check_game_over
Líneas 96 -111  

        _, gray_img = self._get_screenshot()

        res = cv2.matchTemplate(gray_img, self.game_over_template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0:
            return True

Se toma la matriz de la imagen en escala de grises y se utiliza cv2 match template para encontrar un match del 80% de coincidencia al template de game over.  

# step 
Líneas 113 -133  
Ejecuta la acción elegida por el agente y luego calcula el nuevo estado del entorno, la recompensa obtenida y si el juego ha terminado.  

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

Decide la acción a realizar y de ser saltar, presiona space con pyautogui, se detiene 0.01 s para asegurarnos que la acción se registre en el juego.  
Calcula en nuevo vector de observación.  
Establece una recompensa positiva por sobrevivir un paso más.
Verifica si game over es verdadero y de ser así se le da una penalización de 500.  
Por ultimo se le devuelve el formato requerido por gymnasium y Stable-Baselines3.  

# reset
Prepara el entorno después de que el agente choca (o al inicio del entrenamiento), asegurando que el juego esté listo para comenzar.  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pyautogui.press('space')
        observation = self._get_observation()
        info = {}
        return observation, info
        
Llama al método super().reset(seed=seed). Esto asegura que la clase base de Gymnasium (gym.Env) inicialice cualquier estado interno necesario, como la semilla para la generación de números aleatorios si se proporciona.   
Presiona space después de un game over, para pasar de "game over" a "ejecución".  
Captura el estado inicial del juego en el vector de 4, justo después de que el juego comienza.  
Devuelve la observación inicial y un diccionario info vacío.  

#    Entrenamiento.py  
Es el script principal encargado de configurar, iniciar y gestionar el proceso de entrenamiento del agente de Reinforcement Learning (Aprendizaje por Refuerzo) Deep Q-Network (DQN) para jugar al Dino Game.

Su función es orquestar la interacción entre el entorno de juego (DinoEntorno) y el algoritmo de aprendizaje (stable_baselines3.DQN).

# on_preess(key)  
Líneas 12 - 20
    def on_press(key):
        """Activa la señal de salida al pulsar 'q'."""
        global GLOBAL_EXIT_SIGNAL
        try:
            if key.char == 'q':
                GLOBAL_EXIT_SIGNAL = True
                return False  # Detener el listener
        except AttributeError:
            pass

El objetivo principal de esta función es proporcionar una forma segura para que el usuario detenga el entrenamiento del modelo DQN y guarde su progreso sin tener que forzar una interrupción del programa (como con CTRL+C).    

# Init  
Líneas 30 - 34  

     def __init__(self, save_path: str, verbose=0):
            super(GuardarAlPulsarQ, self).__init__(verbose)
            self.save_path = save_path
            self.last_check_time = time.time()
            self.check_interval = 0.5
Inicializa el constructor de la clase GuardaAlPulsarQ al igual que los atributos save_path, last_check y check_interval.  

# _on_step_ 
Líneas 36 - 50  

    def _on_step(self) -> bool:
        global GLOBAL_EXIT_SIGNAL

        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return True
        self.last_check_time = current_time

        if GLOBAL_EXIT_SIGNAL:
            print("\n---TECLA q DETECTADA. GUARDANDO PROGRESO... ---")
            self.model.save(self.save_path)
            print(f"Modelo guardado en: {self.save_path}")
            return False  # Detiene el entrenamiento

        return True

Es invocado por la librería Stable-Baselines3 después de que el agente ejecuta cada paso (step) en el entorno.  
Su propósito principal es verificar periódicamente si el usuario ha presionado la tecla 'q' para detener el entrenamiento de forma segura y guardar el modelo.  

# __name__ == '__main__' 
Líneas 53 - 99 

    if __name__ == '__main__':
        TOTAL_TIMESTEPS = 800_000
        MODEL_SAVE_PATH = "dino_IA.zip"
        NUM_STACK = 4  # 4 frames para inferir la velocidad
    
        # Crear entorno Vectorizado
        env = make_vec_env(DinoEntorno, n_envs=1)
        #Envolver el entorno para apilar 4 frames
        env = VecFrameStack(env, n_stack=NUM_STACK)


Define el número total de pasos en el entorno, especifica el nombre del archivo donde se guardará el modelo entrenado y determina el número de observaciones consecutivas.  
para después crear un entorno vectorizado y el cual envuelve y apila 4 frames.  

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,  # Aumentado 5x para acelerar el aprendizaje
        buffer_size=100_000,
        exploration_fraction=0.1,  # Reducido al 10% para acelerar la explotación
        exploration_final_eps=0.01
    )

Esta es la arquitectura de la red neuronal que nos indica:  
* Utilizamos una política de perceptrón Multicapa.  
* Un valor de 1 indica que se debe imprimir información de progreso durante el entrenamiento (como el número de timesteps y las estadísticas).  
* La tasa de aprendizaje está establecida en 0.0005 y esta controla qué tan grandes son los ajustes realizados a los pesos de la red neuronal después de cada actualización.  
* Define el tamaño del Buffer de Replay (Replay Buffer) 100_000.
* Es la fracción de timesteps totales (en este caso, el $10% de 800,000) durante la cual la exploración decaerá de su valor inicial (1.0) a su valor final (exploration_final_eps).  

       # Callbacks
          interrupt_callback = GuardarAlPulsarQ(save_path=MODEL_SAVE_PATH)
          checkpoint_callback = CheckpointCallback(
              save_freq=10000,
              save_path='./checkpoints/',
              name_prefix='dino_rl_model'
          )
          callbacks_list = [interrupt_callback, checkpoint_callback]
  
Guarda un checkpoint cada 10000 pasos.
  
          print(f"Comenzando el entrenamiento ({TOTAL_TIMESTEPS} pasos). Presiona q o CTRL+C para GUARDAR y salir.")
      
          # Retraso para enfocar la ventana
          print("El entrenamiento comenzará en 3 segundos. Por favor, enfoca la ventana del juego.")
          time.sleep(3)
Comienza el entrenamiento y da 3 segundos para enfocar la pantalla en el juego del dinosaurio.  

          try:
              # Se requiere reset_num_timesteps=True si se cambia el espacio de observación,
              # pero es más seguro eliminar el archivo .zip antes de ejecutar.
              model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks_list, reset_num_timesteps=False)
      
              print("Entrenamiento completado. Guardando el modelo final...")
              model.save(MODEL_SAVE_PATH)
      
          except (KeyboardInterrupt, Exception) as e:
              print(f"\n ERROR o INTERRUPCIÓN. Guardando...")
              model.save(MODEL_SAVE_PATH)


El try de ser exitoso el agente interactuará con el entorno y aprenderá de las recompensas, y actualiza su red neuronal durante el número de pasos específicos, y así el método model.learn se completa (los 800000 se completan el modelo se guarda e imprime un aviso diciendo que se guardó exitosamente).  
En caso de haber alguna interrupción hace un guardado de emergencia para evitar la pérdida de progreso.  

#    JuegaIA 
Es el script de ejecución que te permite cargar el modelo DQN previamente entrenado (dino_IA.zip) e iniciar el juego para que la Inteligencia Artificial juegue al Dino Game de forma autónoma.  

# jugar_con_ia
Líneas 13 - 49  

    print("Iniciando el modo IA. Cargando el modelo entrenado...")

    # Recrear el entorno de la misma forma que durante el entrenamiento
    env = make_vec_env(DinoEntorno, n_envs=1)
    env = VecFrameStack(env, n_stack=NUM_STACK)

    try:
        model = DQN.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f" ERROR: No se pudo cargar el modelo desde {MODEL_PATH}")
        print(f"Detalle del error: {e}")
        return

Recrea el entorno de la misma forma que se creo en el entrenamiento.  
En el bloque try intenta cargar el modelo desde el archivo comprimido en MODEL_PATH.
En except si no encuentra el archivo o esta corrupto usa un return para detener la función.

    # Reset: el entorno apilado devuelve la observación y la información
    obs, info = env.reset()

    try:
        while True:
            # model.predict ya espera el vector apilado (obs)
            action, _states = model.predict(obs, deterministic=True)

            # Ejecutar el paso en el entorno
            obs, reward, done, truncated, info = env.step(action)

            # Si el dinosaurio choca, reiniciamos el juego
            if done[0]:
                print("La IA choco. Reiniciando...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("IA detenida por el usuario.")

    env.close()
Antes de iniciar el bucle, el entorno se reinicia.  
El bucle se ejecuta indefinidamente hasta que el usuario lo detiene (KeyboardInterrupt).
El modelo DQN utiliza la observación apilada (obs) para predecir la mejor acción a tomar (action).  
Asegura que el modelo no realice acciones aleatorias (exploración), sino que siempre elija la acción con el valor Q más alto.  
El entorno actualiza el estado del juego, simula el teclado (pyautogui), y devuelve el nuevo estado (obs), la recompensa, y si el juego terminó (done).  
El agente verifica si el juego ha terminado (si chocó). El índice [0] es necesario porque el entorno es vectorizado (n_envs=1).   
Si choca, se imprime un mensaje y el entorno se reinicia con obs, info = env.reset(), lo que reinicia el juego y obtiene la nueva observación inicial para continuar el bucle.  
Si el usuario presiona CTRL+C en la consola, el bucle se detiene, y se imprime un mensaje de parada.  
Finalmente, se llama a este método para liberar cualquier recurso asociado al entorno

# Cerebro.py
Es un script que implementa una solución de juego reactiva y básica para el Dino Game, utilizando visión por computadora (OpenCV) y simulación de teclado (pyautogui), pero sin emplear ningún algoritmo de Aprendizaje por Refuerzo (IA). 
# detectar_obstaculo_y_saltar
Líneas 14 - 21

    def detectar_obstaculo_y_saltar(binary_img):
        #RECORRE LA LINEA DEL SUELO
        for x in range(X_DINO, X_DINO + DISTANCIA_SALTO):
            #BINARY IMG DEVUELVE EL VALOR DEL PIXEL (255 SI ES SUELO, 0 SI ES OBSTACULO)
            if binary_img[Collision_Y, x] == 0:            #Al detectar un obstaculo salta
                pyautogui.press('space')
                return

Hace un escaneo horizontal para intentar detectar un obstaculo el cuual dara un valor 0 si es un obstaculo y 255 si es el fondo, en caso de encontrar un obstaculo presiona space con pyautogui.  

# procesar_frame
Líneas 25- 42

     with mss.mss() as sct:
            while True:
                sct_img = sct.grab(ROI_DINO_GAME) #Captura el frame en tiempo real
                img = np.array(sct_img) #se convierte en una estructura de datos para el procesamiento con OpenCV (como lo que hicimos con excel)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #lo convierte en escala de grises.

Inicializa la libreria mss para capturar la pantalla y utilizamos el with para liverar los recursos automaticamente al salir del bloque.  
Inicia un bucle infinito que garantiza que el código intente capturar y procesar continuamente frames del juego mientras el script se esté ejecutando.  
Se crea una estructura de datos para el procesamiento con OpenCV y se pasa a escala de grises.

    _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV  )
              if binary_img[deteccionFondo, 140] == 0:
                  _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

Se convierte en binaarios (blancos y negros) y a su vez se invierten los colores, en caso de que el fondo cambie de color se dejan de invertir los colores y se pasa en su estado normal.

    detectar_obstaculo_y_saltar(binary_img)
                cv2.imshow("Obstaculo y saltar", binary_img)
                #Detener el bucle si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
        cv2.destroyAllWindows()
    procesar_frame()

Manda a llamar al detectar_obstaculo_y_saltar y le pasa la imagen en con datos en binarios.  
Imprime la imagen del frame actual y se queda en espera de presionar q para cerrar el programa.

# Recomendaciones para mejor rendimiento de Cerebro.py

comentar líneas: 39, 41, 42, 44.



