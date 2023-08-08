from gym.envs.registration import register

register(
    id="molecule-v0",
    entry_point="envs.MoleculeV0:MoleculeV0",
    max_episode_steps=300
)
