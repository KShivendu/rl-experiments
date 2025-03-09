import gymnasium as gym
from pathlib import Path

from stable_baselines3 import SAC
from time import sleep

ENV_NAME = "HalfCheetah"
MODEL_TYPE = SAC
MODEL_NAME = "sac_cheetah"
LOG_DIR = "logs"

model_path = f"models/{MODEL_NAME}.zip"

env = gym.make(ENV_NAME, render_mode="rgb_array")

if Path(model_path).exists():
    print("Model exists, loading")
    model = MODEL_TYPE.load(model_path, env)
else:
    print("Model doesn't exist")
    model = MODEL_TYPE("MlpPolicy", env, learning_starts=10000, use_sde=False, verbose=1, tensorboard_log=LOG_DIR)

model.learn(total_timesteps=50_000, reset_num_timesteps=False, tb_log_name=MODEL_NAME)
model.save(model_path)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    sleep(0.01)
