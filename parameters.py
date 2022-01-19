import argparse
BOOL_MAP = {
    "true" : True,
    "false" : False
}
parser = argparse.ArgumentParser(
        description='Runs a learning example on a registered gym environment.'
    )
parser.add_argument(
        '--variant',
        default=None,
        type=str
    )
parser.add_argument(
        '--logdir',
        default='test_runs/v0/',
        type=str
    )
def add_controller_args():
    parser.add_argument(
        '--controller_name',
        # Can be 'position', 'position_orientation', 'joint_velocity', 'joint_impedance', or
        #                 'joint_torque'
        default='position_orientation',
        type=str
    )

def add_env_args():
    parser.add_argument(
        '--env',
        default='PandaWipeForce',
        type=str
    )
    parser.add_argument(
        '--env_seed',
        default=17,
        type=int
    )
    parser.add_argument(
        '--use_camera_obs',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--horizon',
        default=1000,
        type=int
    )

def add_train_args():
    parser.add_argument(
        '--alg_seed',
        default=5,
        type=int
    )
    parser.add_argument(
        '--total_timesteps',
        default=1000000,
        type=int
    )