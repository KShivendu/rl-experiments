import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO # A2C,
from time import sleep

ENV_NAME = "HalfCheetah" # HalfCheetah-v5
MODEL_TYPE = PPO
MODEL_NAME = "ppo_cheetah"
LOG_DIR = "logs"

env = gym.make(ENV_NAME, render_mode="rgb_array")

if Path(f"models/{MODEL_NAME}.zip").exists():
    print("Model exists, loading")
    model = MODEL_TYPE.load(f"models/{MODEL_NAME}", env)
else:
    print("Model doesn't exist")
    model = MODEL_TYPE("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)


model.learn(total_timesteps=50_000, reset_num_timesteps=False, tb_log_name=MODEL_NAME)
model.save(MODEL_NAME)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    sleep(0.01)
