import shutil
import numpy as np
import pytest
from gymnasium import spaces
from minari.dataset.step_data import StepData

from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage


def _generate_episode_buffer(observation_space: spaces.Space, action_space: spaces.Space, length=25):
    buffer = EpisodeBuffer(observations=observation_space.sample())

    terminations = np.zeros(length, dtype=np.bool_)
    truncations = np.zeros(length, dtype=np.bool_)
    terminated = np.random.randint(2, dtype=np.bool_)
    terminations[-1] = terminated
    truncations[-1] = not terminated
    rewards = np.random.randn(length)

    for i in range(length):
        action = action_space.sample()
        observation = observation_space.sample()
        step_data: StepData = {
            "observation": observation,
            "action": action,
            "reward": rewards[i],
            "terminated": terminations[i],
            "truncated": truncations[i],
            "info": {},
        }
        buffer = buffer.add_step_data(step_data)
    
    return buffer


@pytest.fixture(params=[
    spaces.Box(-1, 1, shape=(4,)),  # Small observation space
    spaces.Box(-1, 1, shape=(84, 84, 3)),  # Image observation space
    # spaces.Box(-1, 1, shape=(512,)),  # Large observation space
])
def observation_space(request):
    return request.param


@pytest.fixture(params=["hdf5", "arrow"])
def data_format(request):
    return request.param


@pytest.fixture
def action_space():
    return spaces.Box(-1, 1, shape=(1,))


@pytest.fixture
def base_storage_path(tmp_path):
    return str(tmp_path / "minari-benchmark")


@pytest.mark.benchmark
@pytest.mark.parametrize("n_episodes", [10, 100, 1000])
@pytest.mark.parametrize("buffer_length", [32, 256, 1024])
def test_storage_write(benchmark, observation_space, action_space, data_format, base_storage_path, n_episodes, buffer_length):
    def generate_and_write():
        storage = MinariStorage.new(
            data_path=f"{base_storage_path}-{n_episodes}-{buffer_length}",
            observation_space=observation_space,
            action_space=action_space,
            data_format=data_format,
        )
        episodes = [
            _generate_episode_buffer(
                observation_space, action_space, length=buffer_length
            )
            for _ in range(n_episodes)
        ]
        storage.update_episodes(episodes)
        shutil.rmtree(storage.data_path)
    
    benchmark.pedantic(
        generate_and_write,
        rounds=3,
        iterations=1,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_episodes", [100, 1000])
@pytest.mark.parametrize("buffer_length", [256, 2048])
def test_storage_read(benchmark, observation_space, action_space, data_format, base_storage_path, n_episodes, buffer_length):
    # Setup: Create storage and write episodes
    storage = MinariStorage.new(
        data_path=f"{base_storage_path}-read-{n_episodes}-{buffer_length}",
        observation_space=observation_space,
        action_space=action_space,
        data_format=data_format,
    )
    episodes = [
        _generate_episode_buffer(
            observation_space, action_space, length=buffer_length
        )
        for _ in range(n_episodes)
    ]
    storage.update_episodes(episodes)
    
    def read_episodes():
        ep_id = np.random.randint(0, n_episodes)
        storage.get_episodes([ep_id])
    
    benchmark.pedantic(
        read_episodes,
        rounds=3,
        iterations=1,
    )

    # Cleanup
    shutil.rmtree(storage.data_path)
