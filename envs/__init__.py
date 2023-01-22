from gym.envs.registration import register

register(
    id='SumoEnv-v3',
    entry_point='envs.sumo_env_dir:SumoEnv',
)