import gymnasium as  gym 
from stable_baselines3 import PPO

# Load Model
model = PPO.load('cartpole_model.zip')

env   = gym.make('CartPole-v1', render_mode='human')
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, info, _ = env.step(action)
    env.render()
    if done: 
        print('info', info)
        break
    
env.close()