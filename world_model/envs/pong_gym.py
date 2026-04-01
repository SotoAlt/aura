"""PongWorld as a Gymnasium environment for stable-worldmodel training.

Renders 224×224 RGB frames (LeWM ViT-Tiny patch_size=14 expects 224).
Action space: [paddle_left_dy, paddle_right_dy] in [-1, 1].
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from world_model.envs.pong_world import PongWorld


class PongGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array", image_size=224, frameskip=5):
        super().__init__()
        self.pong = PongWorld()
        self.image_size = image_size
        self.render_mode = render_mode
        self.frameskip = frameskip

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            0, 255, shape=(image_size, image_size, 3), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pong.reset(seed=seed)
        obs = self.pong.render(self.image_size)
        return obs, {}

    def step(self, action):
        for _ in range(self.frameskip):
            self.pong.step(action)
        obs = self.pong.render(self.image_size)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"state": self.pong.get_state()}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.pong.render(self.image_size)


# Register with gymnasium
gym.register(
    id="aura/Pong-v1",
    entry_point="world_model.envs.pong_gym:PongGymEnv",
    max_episode_steps=200,
)


class PongAIPolicy:
    """Simple AI policy for data collection — both paddles track ball."""

    def __init__(self, num_envs=1, noise=0.1):
        self.num_envs = num_envs
        self.noise = noise
        self.rng = np.random.default_rng(42)

    def __call__(self, obs):
        # We don't have direct access to ball position from obs (it's an image)
        # This policy is used with the raw env, where we can access pong internals
        # For World.record_dataset, we pass actions manually
        return self.rng.uniform(-1, 1, (self.num_envs, 2)).astype(np.float32)
