import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import data_equivalence

import minari
from minari import DataCollector, MinariDataset
from minari.data_collector import EpisodeBuffer, StepData
from minari.dataset.minari_dataset import EpisodeData
from minari.dataset.minari_storage import MinariStorage


unicode_charset = "".join(
    [chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)) != "Cs"]
)


class DummyBoxEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=4, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=4, shape=(3,), dtype=np.float32
        )

    def _get_info(self):
        return {"timestep": np.array([self.timestep])}

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return (
            self.observation_space.sample(),
            0,
            terminated,
            False,
            self._get_info(),
        )

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.observation_space.seed(seed)
        return self.observation_space.sample(), self._get_info()


class DummyInfoEnv(DummyBoxEnv):
    def __init__(self, info=None):
        super().__init__()
        self.info = info if info is not None else {}

    def _get_info(self):
        return self.info


class DummyInconsistentInfoEnv(DummyBoxEnv):
    def __init__(self):
        super().__init__()

    def _get_info(self):
        return super()._get_info() if self.timestep % 2 == 0 else {}


class DummyMultiDimensionalBoxEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            low=-1, high=4, shape=(2, 2, 2), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1, high=4, shape=(3, 3, 3), dtype=np.float32
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.observation_space.seed(seed)
        return self.observation_space.sample(), {}


class DummyTupleDiscreteBoxEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(1),
                spaces.Discrete(5),
            )
        )
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=-1, high=4, dtype=np.float32),
                spaces.Discrete(5),
            )
        )

    def _get_info(self):
        return {"timestep": np.array([self.timestep])}

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return (
            self.observation_space.sample(),
            0,
            terminated,
            False,
            self._get_info(),
        )

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.observation_space.seed(seed)
        return (
            self.observation_space.sample(),
            self._get_info(),
        )


class DummyDictEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Dict(
            {
                "component_1": spaces.Box(low=-1, high=1, dtype=np.float32),
                "component_2": spaces.Dict(
                    {
                        "subcomponent_1": spaces.Box(low=2, high=3, dtype=np.float32),
                        "subcomponent_2": spaces.Box(low=4, high=5, dtype=np.float32),
                    }
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "component_1": spaces.Box(low=-1, high=1, dtype=np.float32),
                "component_2": spaces.Dict(
                    {
                        "subcomponent_1": spaces.Box(low=2, high=3, dtype=np.float32),
                        "subcomponent_2": spaces.Box(low=4, high=5, dtype=np.float32),
                    }
                ),
            }
        )

    def _get_info(self):
        return {
            "timestep": np.array([self.timestep]),
            "component_1": {"next_timestep": np.array([self.timestep + 1])},
        }

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return (
            self.observation_space.sample(),
            0,
            terminated,
            False,
            self._get_info(),
        )

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.observation_space.seed(seed)

        return self.observation_space.sample(), self._get_info()


class DummyTupleEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Box(low=4, high=5, dtype=np.float32),
            )
        )

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Tuple(
                    (
                        spaces.Box(low=2, high=3, dtype=np.float32),
                        spaces.Box(low=4, high=5, dtype=np.float32),
                    )
                ),
            )
        )

    def _get_info(self):
        return {
            "info_1": np.ones((2, 2)),
            "component_1": {"component_1_info_1": np.ones((2,))},
        }

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, self._get_info()

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.observation_space.seed(seed)

        return self.observation_space.sample(), self._get_info()


class DummyTextEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Text(max_length=10, min_length=2, charset="01")
        self.observation_space = spaces.Text(max_length=20, charset=unicode_charset)

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.observation_space.seed(seed)
        return "者示序袋費欠走立🐝🗓🈸🐿🍯🚆▶️🎧🎇💫", {}


class DummyComboEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Box(low=4, high=5, dtype=np.float32),
            )
        )

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Tuple(
                    (
                        spaces.Box(low=2, high=3, dtype=np.float32),
                        spaces.Dict(
                            {
                                "component_1": spaces.Box(
                                    low=-1, high=1, dtype=np.float32
                                ),
                                "component_2": spaces.Dict(
                                    {
                                        "subcomponent_1": spaces.Box(
                                            low=2, high=3, dtype=np.float32
                                        ),
                                        "subcomponent_2": spaces.Tuple(
                                            (
                                                spaces.Box(
                                                    low=4, high=5, dtype=np.float32
                                                ),
                                                spaces.Discrete(10),
                                            )
                                        ),
                                    }
                                ),
                            }
                        ),
                    )
                ),
            )
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=0, options=None):
        self.timestep = 0
        self.observation_space.seed(seed)
        return self.observation_space.sample(), {}


def register_dummy_envs():
    register(
        id="DummyBoxEnv-v0",
        entry_point="tests.common:DummyBoxEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyInfoEnv-v0",
        entry_point="tests.common:DummyInfoEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyInconsistentInfoEnv-v0",
        entry_point="tests.common:DummyInconsistentInfoEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyMultiDimensionalBoxEnv-v0",
        entry_point="tests.common:DummyMultiDimensionalBoxEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyTupleDiscreteBoxEnv-v0",
        entry_point="tests.common:DummyTupleDiscreteBoxEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyDictEnv-v0",
        entry_point="tests.common:DummyDictEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyTupleEnv-v0",
        entry_point="tests.common:DummyTupleEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyTextEnv-v0",
        entry_point="tests.common:DummyTextEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyComboEnv-v0",
        entry_point="tests.common:DummyComboEnv",
        max_episode_steps=5,
    )


test_spaces = [
    gym.spaces.Box(low=-1, high=4, shape=(2,), dtype=np.float32),
    gym.spaces.Box(low=-1, high=4, shape=(3,), dtype=np.float32),
    gym.spaces.Box(low=-1, high=4, shape=(2, 2, 2), dtype=np.float32),
    gym.spaces.Box(low=-1, high=4, shape=(3, 3, 3), dtype=np.float32),
    gym.spaces.Text(max_length=10, min_length=10),
    gym.spaces.Text(max_length=20, charset=unicode_charset),
    gym.spaces.Text(max_length=10, charset="01"),
    gym.spaces.Tuple(
        (
            gym.spaces.Discrete(1),
            gym.spaces.Discrete(5),
        )
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=-1, high=4, dtype=np.float32),
            gym.spaces.Discrete(5),
        )
    ),
    gym.spaces.Dict(
        {
            "component_1": gym.spaces.Box(low=-1, high=1, dtype=np.float32),
            "component_2": gym.spaces.Dict(
                {
                    "subcomponent_1": gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    "subcomponent_2": gym.spaces.Box(low=4, high=5, dtype=np.float32),
                }
            ),
        }
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=2, high=3, dtype=np.float32),
            gym.spaces.Box(low=4, high=5, dtype=np.float32),
        )
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=2, high=3, dtype=np.float32),
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    gym.spaces.Box(low=4, high=5, dtype=np.float32),
                )
            ),
        )
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=2, high=3, dtype=np.float32),
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    gym.spaces.Dict(
                        {
                            "component_1": gym.spaces.Box(
                                low=-1, high=1, dtype=np.float32
                            ),
                            "component_2": gym.spaces.Dict(
                                {
                                    "subcomponent_1": gym.spaces.Box(
                                        low=2, high=3, dtype=np.float32
                                    ),
                                    "subcomponent_2": gym.spaces.Tuple(
                                        (
                                            gym.spaces.Box(
                                                low=4, high=5, dtype=np.float32
                                            ),
                                            gym.spaces.Discrete(10),
                                        )
                                    ),
                                }
                            ),
                            "component_3": gym.spaces.Text(100, min_length=20),
                        }
                    ),
                )
            ),
        )
    ),
]
unsupported_test_spaces = [
    gym.spaces.Graph(
        gym.spaces.Box(low=-1, high=4, shape=(3,), dtype=np.float32), None
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=2, high=3, dtype=np.float32),
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    gym.spaces.Dict(
                        {
                            "component_1": gym.spaces.Box(
                                low=-1, high=1, dtype=np.float32
                            ),
                            "component_2": gym.spaces.Dict(
                                {
                                    "subcomponent_1": gym.spaces.Box(
                                        low=2, high=3, dtype=np.float32
                                    ),
                                    "subcomponent_2": gym.spaces.Tuple(
                                        (
                                            gym.spaces.Box(
                                                low=4, high=5, dtype=np.float32
                                            ),
                                            gym.spaces.Graph(
                                                gym.spaces.Box(-1, 1), None
                                            ),
                                        )
                                    ),
                                }
                            ),
                        }
                    ),
                )
            ),
        )
    ),
]


def check_env_recovery_with_subset_spaces(
    gymnasium_environment: gym.Env,
    dataset: MinariDataset,
    action_space_subset: gym.spaces.Space,
    observation_space_subset: gym.spaces.Space,
):
    """Test that the recovered environment from MinariDataset is the same as the one used to generate the dataset.

    Args:
        gymnasium_environment (gym.Env): original Gymnasium environment
        dataset (MinariDataset): Minari dataset created with gymnasium_environment
        action_space_subset (gym.spaces.Space): desired subset action space
        observation_space_subset (gym.spaces.Space): desired subset observation space

    """
    recovered_env = dataset.recover_environment()

    # Check that environment spec is the same
    assert (
        recovered_env.spec == gymnasium_environment.spec
    ), f"recovered_env spec: {recovered_env.spec}\noriginal spec: {gymnasium_environment.spec}"

    # Check that action/observation spaces are the same
    assert data_equivalence(
        recovered_env.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(dataset.spec.observation_space, observation_space_subset)
    assert data_equivalence(
        recovered_env.action_space, gymnasium_environment.action_space
    )
    assert data_equivalence(dataset.spec.action_space, action_space_subset)


def check_env_recovery(
    gymnasium_environment: gym.Env,
    dataset: MinariDataset,
    evaluation_environment: Optional[gym.Env] = None,
):
    """Test that the recovered environment from MinariDataset is the same as the one used to generate the dataset.

    Args:
        gymnasium_environment (gym.Env): original Gymnasium environment used to create the dataset.
        dataset (MinariDataset): Minari dataset created with gymnasium_environment
        evaluation_environment (gym.Env): Gymnasium environment saved in the `eval_env` attribute of the MinariDataset that should be used for evaluation. This attribute is optional.
    """
    recovered_env = dataset.recover_environment()

    # Check that environment spec is the same
    assert (
        recovered_env.spec == gymnasium_environment.spec
    ), f"recovered_env spec: {recovered_env.spec}\noriginal spec: {gymnasium_environment.spec}"

    # Check that action/observation spaces are the same
    assert data_equivalence(
        recovered_env.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(
        dataset.spec.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(
        recovered_env.action_space, gymnasium_environment.action_space
    )
    assert data_equivalence(
        dataset.spec.action_space, gymnasium_environment.action_space
    )

    if evaluation_environment is not None:
        recovered_eval_env = dataset.recover_environment(eval_env=True)

        # Check that evaluation environment spec is the same
        assert (
            recovered_eval_env.spec == evaluation_environment.spec
        ), f"recovered_eval_env spec: {recovered_eval_env.spec}\noriginal spec: {evaluation_environment}"


def check_data_integrity(data: MinariStorage, episode_indices: Iterable[int]):
    """Checks to see if a MinariStorage episode has consistent data and has episodes at the expected indices.

    Args:
        data (MinariStorage): a MinariStorage instance
        episode_indices (Iterable[int]): the list of episode indices expected
    """
    episodes = list(data.get_episodes(episode_indices))
    # verify we have the right number of episodes, available at the right indices
    assert data.total_episodes == len(
        episodes
    ), f"{data.total_episodes} != {len(episodes)}"
    total_steps = 0

    observation_space = data.metadata["observation_space"]
    action_space = data.metadata["action_space"]

    # verify the actions and observations are in the appropriate action space and observation space, and that the episode lengths are correct
    for episode in episodes:
        total_steps += episode["total_steps"]
        _check_space_elem(
            episode["observations"],
            observation_space,
            episode["total_steps"] + 1,
        )
        _check_space_elem(episode["actions"], action_space, episode["total_steps"])

        for i in range(episode["total_steps"] + 1):
            obs = _reconstuct_obs_or_action_at_index_recursive(
                episode["observations"], i
            )
            assert observation_space.contains(obs)
        for i in range(episode["total_steps"]):
            action = _reconstuct_obs_or_action_at_index_recursive(episode["actions"], i)
            assert action_space.contains(action)

        assert episode["total_steps"] == len(episode["rewards"])
        assert episode["total_steps"] == len(episode["terminations"])
        assert episode["total_steps"] == len(episode["truncations"])

    assert total_steps == data.total_steps


def get_info_at_step_index(infos: Dict, step_index: int) -> Dict:
    result = {}
    for key in infos.keys():
        if isinstance(infos[key], dict):
            result[key] = get_info_at_step_index(infos[key], step_index)
        elif isinstance(infos[key], np.ndarray):
            result[key] = infos[key][step_index]
        else:
            raise ValueError(
                "Infos are in an unsupported format; see Minari documentation for supported formats."
            )
    return result


def _reconstuct_obs_or_action_at_index_recursive(
    data: Union[dict, tuple, np.ndarray], index: int
) -> Union[np.ndarray, dict, tuple]:
    if isinstance(data, dict):
        return {
            key: _reconstuct_obs_or_action_at_index_recursive(data[key], index)
            for key in data.keys()
        }
    elif isinstance(data, tuple):
        return tuple(
            [
                _reconstuct_obs_or_action_at_index_recursive(entry, index)
                for entry in data
            ]
        )
    else:
        assert isinstance(
            data, (np.ndarray, List)
        ), "error, invalid observation or action structure"
        return data[index]


def _check_space_elem(data: Any, space: spaces.Space, n_elements: int):
    if isinstance(space, spaces.Tuple):
        assert isinstance(data, tuple)
        assert len(data) == len(space.spaces)
        for data_elem, sub_space in zip(data, space.spaces):
            _check_space_elem(data_elem, sub_space, n_elements)
    elif isinstance(space, spaces.Dict):
        assert isinstance(data, dict)
        assert data.keys() == space.keys()
        for key in data.keys():
            _check_space_elem(data[key], space[key], n_elements)
    else:
        assert len(data) == n_elements


def check_load_and_delete_dataset(dataset_id: str):
    """Test loading and deletion of local Minari datasets.

    Args:
        dataset_id (str): name of Minari dataset to test
    """
    # check dataset name is present in local database
    local_datasets = minari.list_local_datasets()
    assert dataset_id in local_datasets

    # load dataset
    loaded_dataset = minari.load_dataset(dataset_id)
    assert isinstance(loaded_dataset, MinariDataset)
    assert dataset_id == loaded_dataset.spec.dataset_id

    # delete dataset and check that it's no longer present in local database
    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets


def create_dummy_dataset_with_collecter_env_helper(
    dataset_id: str, env: DataCollector, num_episodes: int = 10, **kwargs
):
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    # Step the environment, DataCollector wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    # Create Minari dataset and store locally
    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/tests/common.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
        **kwargs,
    )

    assert dataset_id in minari.list_local_datasets()
    return dataset


def check_episode_data_integrity(
    episode_data_list: Union[List[EpisodeData], MinariDataset],
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    info_sample: Optional[dict] = None,
):
    """Checks to see if a list of EpisodeData instances has consistent data and that the observations and actions are in the appropriate spaces.

    Args:
        episode_data_list (List[EpisodeData]): A list of EpisodeData instances representing episodes.
        observation_space (gym.spaces.Space): The environment's observation space.
        action_space (gym.spaces.Space): The environment's action space.
        info_sample (dict): An info returned by the environment used to build the dataset.

    """
    # verify the actions and observations are in the appropriate action space and observation space, and that the episode lengths are correct
    for episode in episode_data_list:
        _check_space_elem(
            episode.observations,
            observation_space,
            episode.total_steps + 1,
        )
        _check_space_elem(episode.actions, action_space, episode.total_steps)

        for i in range(episode.total_steps + 1):
            obs = _reconstuct_obs_or_action_at_index_recursive(episode.observations, i)
            if info_sample is not None:
                assert episode.infos is not None
                assert check_infos_equal(
                    get_info_at_step_index(episode.infos, i), info_sample
                )

            assert observation_space.contains(obs)

        for i in range(episode.total_steps):
            action = _reconstuct_obs_or_action_at_index_recursive(episode.actions, i)
            assert action_space.contains(action)

        assert episode.total_steps == len(episode.rewards)
        assert episode.total_steps == len(episode.terminations)
        assert episode.total_steps == len(episode.truncations)


def check_infos_equal(info_1: Dict, info_2: Dict) -> bool:
    if info_1.keys() != info_2.keys():
        return False
    for key in info_1.keys():
        if isinstance(info_1[key], dict):
            return check_infos_equal(info_1[key], info_2[key])
        elif isinstance(info_1[key], np.ndarray):
            return bool(np.all(info_1[key] == info_2[key]))
        else:
            return info_1[key] == info_2[key]
    return True


def _space_subset_helper(entry: Dict):
    return {"component_2": {"subcomponent_2": entry["component_2"]["subcomponent_2"]}}


def get_sample_buffer_for_dataset_from_env(env: gym.Env, num_episodes: int = 10):
    buffer = []
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
                "observations": observation,
                "actions": action,
                "rewards": reward,
                "terminations": terminated,
                "truncations": truncated,
                "infos": {},
            }
            episode_buffer = episode_buffer.add_step_data(step_data)

        buffer.append(episode_buffer)

        observation, _ = env.reset()
        episode_buffer = EpisodeBuffer(observations=observation)

    return buffer
