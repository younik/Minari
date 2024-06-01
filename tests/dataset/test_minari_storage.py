import os
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import minari
from minari import DataCollector
from minari.data_collector.callbacks.step_callback import StepData
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset._storages import registry as storage_registry
from minari.dataset.minari_storage import MinariStorage
from tests.common import check_data_integrity, check_load_and_delete_dataset


file_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")


def _generate_episode_buffer(
    observation_space: spaces.Space, action_space: spaces.Space, length=25
):
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
            "termination": terminations[i],
            "truncation": truncations[i],
            "info": {},
        }
        buffer = buffer.add_step_data(step_data)

    return buffer


def test_non_existing_data(tmp_dataset_dir):
    with pytest.raises(ValueError, match="The data path foo doesn't exist"):
        MinariStorage.read("foo")

    with pytest.raises(ValueError, match="No data found in data path"):
        MinariStorage.read(tmp_dataset_dir)


@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_metadata(tmp_dataset_dir, data_format):
    action_space = spaces.Box(-1, 1)
    observation_space = spaces.Box(-1, 1)
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
        data_format=data_format,
    )
    assert str(storage.data_path) == tmp_dataset_dir

    extra_metadata = {"float": 3.2, "string": "test-value", "int": 2, "bool": True}
    storage.update_metadata(extra_metadata)

    storage_metadata = storage.metadata
    assert storage_metadata.keys() == {
        "action_space",
        "bool",
        "float",
        "int",
        "observation_space",
        "string",
        "total_episodes",
        "total_steps",
        "data_format",
    }

    for key, value in extra_metadata.items():
        assert storage_metadata[key] == value

    storage2 = MinariStorage.read(tmp_dataset_dir)
    assert storage_metadata == storage2.metadata


@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_add_episodes(tmp_dataset_dir, data_format):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    steps_per_episode = 25
    episodes = [
        _generate_episode_buffer(
            observation_space, action_space, length=steps_per_episode
        )
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
        data_format=data_format,
    )
    storage.update_episodes(episodes)
    del storage
    storage = MinariStorage.read(tmp_dataset_dir)

    assert storage.total_episodes == n_episodes
    assert storage.total_steps == n_episodes * steps_per_episode

    for i, ep in enumerate(episodes):
        storage_ep = storage.get_episodes([i])[0]
        assert np.all(ep.observations == storage_ep["observations"])
        assert np.all(ep.actions == storage_ep["actions"])
        assert np.all(ep.rewards == storage_ep["rewards"])
        assert np.all(ep.terminations == storage_ep["terminations"])
        assert np.all(ep.truncations == storage_ep["truncations"])


@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_apply(tmp_dataset_dir, data_format):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    episodes = [
        _generate_episode_buffer(observation_space, action_space)
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
        data_format=data_format,
    )
    storage.update_episodes(episodes)

    def f(ep):
        return ep["actions"].sum()

    episode_indices = [1, 3, 5]
    outs = storage.apply(f, episode_indices=episode_indices)
    assert len(episode_indices) == len(list(outs))
    for i, result in zip(episode_indices, outs):
        assert np.array(episodes[i].actions).sum() == result


@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_episode_metadata(tmp_dataset_dir, data_format):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    episodes = [
        _generate_episode_buffer(observation_space, action_space)
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
        data_format=data_format,
    )
    storage.update_episodes(episodes)

    ep_metadatas = [
        {"foo1-1": True, "foo1-2": 7},
        {"foo2-1": 3.14},
        {"foo3-1": "foo", "foo3-2": 42, "foo3-3": "test"},
    ]

    ep_indices = [1, 4, 5]
    storage.update_episode_metadata(ep_metadatas, episode_indices=ep_indices)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_minari_get_dataset_size_from_collector_env(
    dataset_id, env_id, data_format, register_dummy_envs
):
    """Test get_dataset_size method for dataset made with DataCollector environment."""
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollector(env, data_format=data_format)
    num_episodes = 100

    # Step the environment, DataCollector wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        done = False
        while not done:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.reset()

    # Create Minari dataset and store locally
    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert dataset.storage.metadata["dataset_size"] == dataset.storage.get_size()

    check_data_integrity(dataset.storage, dataset.episode_indices)

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-text-test-v0", "DummyTextEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_minari_get_dataset_size_from_buffer(
    dataset_id, env_id, data_format, register_dummy_envs
):
    """Test get_dataset_size method for dataset made using create_dataset_from_buffers method."""
    buffer = []

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    num_episodes = 10
    seed = 42
    observation, _ = env.reset(seed=seed)
    episode_buffer = EpisodeBuffer(observations=observation, seed=seed)

    for episode in range(num_episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            step_data: StepData = {
                "observation": observation,
                "action": action,
                "reward": reward,
                "termination": terminated,
                "truncation": truncated,
                "info": {},
            }
            episode_buffer = episode_buffer.add_step_data(step_data)

        buffer.append(episode_buffer)

        observation, _ = env.reset()
        episode_buffer = EpisodeBuffer(observations=observation)

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=buffer,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
        data_format=data_format,
    )

    assert dataset.storage.metadata["dataset_size"] == dataset.storage.get_size()

    check_data_integrity(dataset.storage, dataset.episode_indices)

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_seed_change(tmp_dataset_dir, data_format):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Discrete(10)
    episodes = []
    seeds = [None, 42]
    for seed in seeds:
        ep = _generate_episode_buffer(observation_space, action_space)
        episodes.append(replace(ep, seed=seed))

    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
        data_format=data_format,
    )
    storage.update_episodes(episodes)

    assert storage.total_episodes == len(seeds)
    episodes = storage.get_episodes(range(len(episodes)))
    assert len(episodes) == len(seeds)
    for seed, ep in zip(seeds, episodes):
        assert ep["seed"] == seed
