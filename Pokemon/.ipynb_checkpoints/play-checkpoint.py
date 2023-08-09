import os
import gym
import retro
import numpy as np
import time
from gym import Env
from gym.spaces import MultiBinary, Box
import cv2
from matplotlib import pyplot as plt
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
LOG_DIR = './logsLegacy/'
OPT_DIR = './opt1/'
OPT0_DIR = './optLegacy/'
CHECKPOINT_DIR = './checkpoints/'

class TetrisAttackCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path, verbose=1):
        super(TetrisAttackCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls)))
        return True
    


class TetrisAttack(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='TetrisAttack-Snes' , use_restricted_actions=retro.Actions.FILTERED)
        # env = retro.make(game='TetrisAttack-Snes' , use_restricted_actions=retro.Actions.FILTERED) 
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)

        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        reward = info['score'] - self.score
        self.score = info['score']

        return frame_delta, reward, done, info
    def render(self, *args, **kwargs):
        self.game.render()
    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs
    def preprocess(self,observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels
    def close(self):
        self.game.close()

def optimize_ppo(trial):
    return {
        'n_steps':trial.suggest_int('n_steps', 2048, 8192),
        'gamma':trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, .99),
    }

def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)

        env = TetrisAttack()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=10)

        
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        env.close()
        
        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward
    except Exception as e:
        print(e)
        return -1000


# study = optuna.create_study(direction='maximize')
# study.optimize(optimize_agent, n_trials=1, n_jobs=1)

# bestModel = study.best_trial
# print(bestModel)

# study = optuna.study.load_study(study_name='trial_14_best_model', storage='sqlite:///opt0.db')

# print(model.summary())

# model_params = study.best_params
callback = TetrisAttackCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model = PPO.load(os.path.join(OPT0_DIR, 'trial_14_best_model'))
model_params = dict(
    n_steps= model.n_steps,
    gamma= model.gamma,
    learning_rate= model.learning_rate,
    clip_range= model.clip_range,
    gae_lambda= model.gae_lambda,
)
# print(model_params)


env = TetrisAttack()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='first')

model_params['n_steps'] = (model_params['n_steps']%64) * 64


# print(model_params)

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
model = PPO.load(os.path.join(OPT0_DIR, 'trial_14_best_model.zip'))
model.learn(total_timesteps=1000000, callback=callback)





# {'n_steps': 4433, 'gamma': 0.9446568952906825, 'learning_rate': 2.0693404659710535e-05, 'clip_range': 0.29576505444633994, 'gae_lambda': 0.8215565190231281}
