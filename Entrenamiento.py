from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from pynput import keyboard
import time
from DinoEntorno import DinoEntorno

GLOBAL_EXIT_SIGNAL = False


def on_press(key):
    """Activa la señal de salida al pulsar 'q'."""
    global GLOBAL_EXIT_SIGNAL
    try:
        if key.char == 'q':
            GLOBAL_EXIT_SIGNAL = True
            return False  # Detener el listener
    except AttributeError:
        pass


# Iniciar el listener de teclado
listener = keyboard.Listener(on_press=on_press)
listener.daemon = True
listener.start()


class GuardarAlPulsarQ(BaseCallback):
    def __init__(self, save_path: str, verbose=0):
        super(GuardarAlPulsarQ, self).__init__(verbose)
        self.save_path = save_path
        self.last_check_time = time.time()
        self.check_interval = 0.5

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


if __name__ == '__main__':
    TOTAL_TIMESTEPS = 800_000
    MODEL_SAVE_PATH = "dino_IA.zip"
    NUM_STACK = 4  # 4 frames para inferir la velocidad

    # Crear entorno Vectorizado
    env = make_vec_env(DinoEntorno, n_envs=1)
    #Envolver el entorno para apilar 4 frames
    env = VecFrameStack(env, n_stack=NUM_STACK)

    # Crear el Modelo
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,  # Aumentado 5x para acelerar el aprendizaje
        buffer_size=100_000,
        exploration_fraction=0.1,  # Reducido al 10% para acelerar la explotación
        exploration_final_eps=0.01
    )

    # Callbacks
    interrupt_callback = GuardarAlPulsarQ(save_path=MODEL_SAVE_PATH)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='dino_rl_model'
    )
    callbacks_list = [interrupt_callback, checkpoint_callback]

    print(f"Comenzando el entrenamiento ({TOTAL_TIMESTEPS} pasos). Presiona q o CTRL+C para GUARDAR y salir.")

    # Retraso para enfocar la ventana
    print("El entrenamiento comenzará en 3 segundos. Por favor, enfoca la ventana del juego.")
    time.sleep(3)

    try:
        # Se requiere reset_num_timesteps=True si se cambia el espacio de observación,
        # pero es más seguro eliminar el archivo .zip antes de ejecutar.
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks_list, reset_num_timesteps=False)

        print("Entrenamiento completado. Guardando el modelo final...")
        model.save(MODEL_SAVE_PATH)

    except (KeyboardInterrupt, Exception) as e:
        print(f"\n ERROR o INTERRUPCIÓN. Guardando...")
        model.save(MODEL_SAVE_PATH)