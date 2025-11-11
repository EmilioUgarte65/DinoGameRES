from stable_baselines3 import DQN
import time

from DinoEntorno import DinoEntorno
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# ruta del modelo
MODEL_PATH = "dino_IA.zip"
NUM_STACK = 4


def jugar_con_ia():
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

    print("Modelo cargado. Presiona una tecla para enfocar la ventana y empezar.")
    time.sleep(3)

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


if __name__ == '__main__':
    jugar_con_ia()