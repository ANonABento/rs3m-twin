import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rs3m_twin.env  # noqa: F401  (registers RS3M-v0)
from rs3m_twin import invert_moves
from rs3m_twin.env import RS3MEnv


def test_check_env():
    check_env(gym.make("RS3M-v0", magnet_model="sin4").unwrapped, skip_render_check=True)


def test_seeds_give_different_physics():
    env = RS3MEnv(magnet_model="sin4")
    env.reset(seed=1)
    p1 = env.sim.params
    env.reset(seed=2)
    p2 = env.sim.params
    assert p1.friction != p2.friction and p1.magnet_k != p2.magnet_k
    env.reset(seed=1)
    assert env.sim.params == p1


def test_random_policy_no_nans():
    env = RS3MEnv(magnet_model="sin4", max_steps=10_000)
    env.action_space.seed(0)
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            obs, _ = env.reset()


@pytest.mark.parametrize("mechanism,magnet_model", [("torque", "dipole"), ("torque", "sin4")])
def test_inverse_scramble_solves(mechanism, magnet_model):
    env = RS3MEnv(mechanism=mechanism, magnet_model=magnet_model, scramble_depth=4)
    _, info = env.reset(seed=3)
    assert not info["solved"]
    env.sim.apply_moves(invert_moves(info["scramble"]))
    _, r, term, _, info = env.step(np.zeros(6))
    assert info["solved"] and term and r == 1.0
    assert env.scramble_depth == 5  # curriculum
