import json
import os
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import robosuite as suite
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def run_learn(params, save_path=''):
    set_seed(params["env_params"]["env_seed"])
    # load controller
    controller_config_file = 'robosuite/controllers/controller_config.hjson'
    controller_config = params.get("controller_params", None)
    if params["env_params"]["use_camera_obs"]:
        # learning on pixels
        env = suite.make(
            params["env_params"]["env"],
            has_renderer=False,
            has_offscreen_render=True,
            ignore_done=False,
            use_camera_obs=True,
            camera_height=48,
            camera_width=48,
            camera_name='agentview',
            use_object_obs=True,
            reward_shaping=False,
            control_freq=20,
            controller= params["controller_params"]["controller_name"],
            horizon=params["env_params"]["horizon"],
            controller_config_file=controller_config_file,
            **controller_config
        )
        eval_env = suite.make(
            params["env_params"]["env"],
            has_renderer=False,
            has_offscreen_render=True,
            ignore_done=False,
            use_camera_obs=True,
            camera_height=48,
            camera_width=48,
            camera_name='agentview',
            use_object_obs=True,
            reward_shaping=False,
            control_freq=20,
            controller=params["controller_params"]["controller_name"],
            horizon=params["env_params"]["horizon"],
            controller_config_file=controller_config_file,
            **controller_config
        )
    else:
        # learning on states
        env = suite.make(
            params["env_params"]["env"],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_object_obs=True,
            use_camera_obs=False,
            control_freq=20,
            controller=params["controller_params"]["controller_name"],
            horizon=params["env_params"]["horizon"],
            controller_config_file=controller_config_file,
            **controller_config
        )
        eval_env = suite.make(
            params["env_params"]["env"],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_object_obs=True,
            use_camera_obs=False,
            control_freq=20,
            controller=params["controller_params"]["controller_name"],
            horizon=params["env_params"]["horizon"],
            controller_config_file=controller_config_file,
            **controller_config
        )
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    from robosuite.wrappers.gym_wrapper import GymWrapper
    env = GymWrapper(env, logdir=save_path)
    eval_env = Monitor(GymWrapper(eval_env))
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path, log_path=save_path, eval_freq=10000)
    model = PPO(
        'MlpPolicy',
        env,
        seed=params["train_args"]["alg_seed"],
    )
    model.learn(total_timesteps=params["train_args"]["total_timesteps"],callback=eval_callback)
    model.save(os.path.join(save_path, "model.zip"))

if __name__ == '__main__':
    import mujoco_py
    import glfw
    from parameters import *
    add_env_args()
    add_train_args()
    add_controller_args()
    args = parser.parse_args()

    if args.variant is None:
        params = dict(
            controller_params=dict(
                controller_name=args.controller_name,
            ),
            env_params=dict(
                env=args.env,
                env_seed=args.env_seed,
                use_camera_obs=args.use_camera_obs,
                horizon=args.horizon,
            ),
            train_args=dict(
                alg_seed=args.alg_seed,
                total_timesteps=args.total_timesteps,
            ),
        )
        save_dir = os.path.join(args.logdir, "{}_{}_{}_horizon{}_seed{}".format(args.env,
                                                      args.controller_name,
                                                      args.use_camera_obs,
                                                      args.horizon,
                                                      args.env_seed))
    run_learn(params=params, save_path=save_dir)


