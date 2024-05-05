from __future__ import annotations

import pathlib
from collections import OrderedDict
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np

from minari.data_collector import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage


_MAIN_FILE_NAME = "main_data.hdf5"


class HDF5Storage(MinariStorage):
    FORMAT = "hdf5"

    def __init__(
        self,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        super().__init__(data_path, observation_space, action_space)
        file_path = self.data_path.joinpath(_MAIN_FILE_NAME)
        if not file_path.exists():
            raise ValueError(f"No data found in data path {self.data_path}")
        self._file_path = file_path

    @classmethod
    def _create(
        cls,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> MinariStorage:
        data_path.joinpath(_MAIN_FILE_NAME).touch(exist_ok=False)
        obj = cls(data_path, observation_space, action_space)
        return obj

    def update_episode_metadata(
        self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None
    ):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)

        sentinel = object()
        with h5py.File(self._file_path, "a") as file:
            for metadata, episode_id in zip_longest(
                metadatas, episode_indices, fillvalue=sentinel
            ):
                if sentinel in (metadata, episode_id):
                    raise ValueError(
                        "Metadatas and episode_indices have different lengths"
                    )

                assert isinstance(metadata, dict)
                ep_group = file[f"episode_{episode_id}"]
                ep_group.attrs.update(metadata)

    def _decode_space(
        self,
        hdf_ref: Union[h5py.Group, h5py.Dataset, h5py.Datatype],
        space: gym.spaces.Space,
    ) -> Union[Dict, Tuple, List, np.ndarray]:
        assert not isinstance(hdf_ref, h5py.Datatype)

        if isinstance(space, gym.spaces.Tuple):
            assert isinstance(hdf_ref, h5py.Group)
            result = []
            for i in range(len(hdf_ref.keys())):
                result.append(
                    self._decode_space(hdf_ref[f"_index_{i}"], space.spaces[i])
                )
            return tuple(result)
        elif isinstance(space, gym.spaces.Dict):
            assert isinstance(hdf_ref, h5py.Group)
            result = {}
            for key in hdf_ref.keys():
                result[key] = self._decode_space(hdf_ref[key], space.spaces[key])
            return result
        elif isinstance(space, gym.spaces.Text):
            assert isinstance(hdf_ref, h5py.Dataset)
            result = map(lambda string: string.decode("utf-8"), hdf_ref[()])
            return list(result)
        else:
            assert isinstance(hdf_ref, h5py.Dataset)
            return hdf_ref[()]

    def _decode_infos(self, infos: h5py.Group):
        result = {}
        for key, value in infos.items():
            if isinstance(value, h5py.Group):
                result[key] = self._decode_infos(value)
            elif isinstance(value, h5py.Dataset):
                result[key] = value[()]
            else:
                raise ValueError(
                    "Infos are in an unsupported format; see Minari documentation for supported formats."
                )
        return result

    def get_episodes(self, episode_indices: Iterable[int]) -> List[dict]:
        outs = []
        with h5py.File(self._file_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)

                seed = ep_group.attrs.get("seed")
                if isinstance(seed, np.integer):
                    seed = int(seed)
                infos = None
                if "infos" in ep_group:
                    info_group = ep_group["infos"]
                    assert isinstance(info_group, h5py.Group)
                    infos = self._decode_infos(info_group)

                ep_dict = {
                    "id": ep_group.attrs.get("id"),
                    "total_steps": ep_group.attrs.get("total_steps"),
                    "seed": seed,
                    "observations": self._decode_space(
                        ep_group["observations"], self.observation_space
                    ),
                    "actions": self._decode_space(
                        ep_group["actions"], self.action_space
                    ),
                    "infos": infos,
                }
                for key in {"rewards", "terminations", "truncations"}:
                    group_value = ep_group[key]
                    assert isinstance(group_value, h5py.Dataset)
                    ep_dict[key] = group_value[:]

                outs.append(ep_dict)

        return outs

    def update_episodes(self, episodes: Iterable[EpisodeBuffer]):
        additional_steps = 0
        with h5py.File(self._file_path, "a", track_order=True) as file:
            for eps_buff in episodes:
                total_episodes = len(file.keys())
                episode_id = eps_buff.id if eps_buff.id is not None else total_episodes
                assert (
                    episode_id <= total_episodes
                ), "Invalid episode id; ids must be sequential."
                episode_group = _get_from_h5py(file, f"episode_{episode_id}")
                episode_group.attrs["id"] = episode_id
                if eps_buff.seed is not None:
                    assert "seed" not in episode_group.attrs.keys()
                    episode_group.attrs["seed"] = eps_buff.seed
                episode_steps = len(eps_buff.rewards)
                episode_group.attrs["total_steps"] = episode_steps
                additional_steps += episode_steps

                dict_buffer = {
                    "observations": eps_buff.observations,
                    "actions": eps_buff.actions,
                    "rewards": eps_buff.rewards,
                    "terminations": eps_buff.terminations,
                    "truncations": eps_buff.truncations,
                    "infos": eps_buff.infos,
                }
                _add_episode_to_group(dict_buffer, episode_group)

            total_episodes = len(file.keys())

        total_steps = self.total_steps + additional_steps
        self.update_metadata(
            {"total_steps": total_steps, "total_episodes": total_episodes}
        )


def _get_from_h5py(group: h5py.Group, name: str) -> h5py.Group:
    if name in group:
        subgroup = group.get(name)
        assert isinstance(subgroup, h5py.Group)
    else:
        subgroup = group.create_group(name)

    return subgroup


def _add_episode_to_group(episode_buffer: Dict, episode_group: h5py.Group):
    for key, data in episode_buffer.items():
        if isinstance(data, dict):
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(data, episode_group_to_clear)
        elif isinstance(data, tuple):
            dict_data = {f"_index_{i}": subdata for i, subdata in enumerate(data)}
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)
        elif all(
            isinstance(entry, OrderedDict) for entry in data
        ):  # list of OrderedDict
            dict_data = {key: [entry[key] for entry in data] for key in data[0].keys()}
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)

        # leaf data
        elif key in episode_group:
            dataset = episode_group[key]
            assert isinstance(dataset, h5py.Dataset)
            dataset.resize((dataset.shape[0] + len(data), *dataset.shape[1:]))
            dataset[-len(data) :] = data
        else:
            dtype = None
            if all(map(lambda elem: isinstance(elem, str), data)):
                dtype = h5py.string_dtype(encoding="utf-8")
            dshape = ()
            if hasattr(data[0], "shape"):
                dshape = data[0].shape

            episode_group.create_dataset(
                key, data=data, dtype=dtype, chunks=True, maxshape=(None, *dshape)
            )
