from __future__ import annotations
import json

import pathlib
from itertools import zip_longest
from typing import Any, Dict, Iterable, List, Optional, Sequence
import gymnasium as gym
import pyarrow as pa
import pyarrow.dataset as ds
from minari.dataset.minari_storage import MinariStorage
import numpy as np


class ArrowStorage(MinariStorage):
    def __init__(self, data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space
    ):
        super().__init__(data_path, observation_space, action_space)

    @classmethod
    def _create(cls, data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space
    ) -> MinariStorage:
        return cls(data_path, observation_space, action_space)

    def update_episode_metadata(
        self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None
    ):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)

        sentinel = object()
        for new_metadata, episode_id in zip_longest(metadatas, episode_indices, fillvalue=sentinel):
            if sentinel in (new_metadata, episode_id):
                raise ValueError('Metadatas and episode_indices have different lengths')

            assert isinstance(new_metadata, dict)
            metadata_path = self.data_path.joinpath(str(episode_id), "metadata.json")
            
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r") as file:
                    metadata = json.load(file)
            metadata.update(new_metadata)
            with open(metadata_path, "w") as file:
                json.dump(metadata, file)

    def get_episodes(self, episode_indices: Iterable[int]) -> List[dict]:
        episode_indices = list(episode_indices)
        dataset = pa.dataset.dataset(
            self.data_path,
            format="parquet",
            partitioning=["episode_id"],
            ignore_prefixes=["_", ".", "metadata.json"]
        )
        episodes = dataset.filter(pa.compute.field("episode_id").isin(episode_indices))

        def _to_dict(episode):
            return {
                "id": episode["episode_id"][0].as_py(),
                "seed": episode["seed"][0].as_py() if "seed" in episode.column_names else None,
                "total_steps": len(episode) - 1,
                "observations": _decode_space(self.observation_space, episode["observations"]),
                "actions": _decode_space(self.action_space, episode["actions"][:-1]),
                "rewards": np.asarray(episode["rewards"])[:-1],
                "terminations": np.asarray(episode["terminations"])[:-1],
                "truncations": np.asarray(episode["truncations"])[:-1],
                "infos": _decode_info(episode["infos"]) if "infos" in episode.column_names else {},
            }
        episodes = map(_to_dict, episodes.to_batches())
        return list(episodes)


    def update_episodes(self, episodes: Iterable[dict]):
        total_steps = self.total_steps
        total_episodes = self.total_episodes
        for episode_data in episodes:
            episode_id = episode_data.get("id", total_episodes)
            total_episodes = max(total_episodes, episode_id + 1)
            observations= _encode_space(self.observation_space, episode_data["observations"])
            rewards = np.asarray(episode_data["rewards"]).reshape(-1)
            terminations = np.asarray(episode_data["terminations"]).reshape(-1)
            truncations = np.asarray(episode_data["truncations"]).reshape(-1)
            pad = len(observations) - len(rewards)  # MULTIPLE STORES SAME EP; PAD MULTIPLES?
            actions = _encode_space(self._action_space, episode_data["actions"], pad=pad)

            episode_batch = {
                "episode_id": np.full(len(observations), episode_id, dtype=np.int32),
                "observations": observations,
                "actions": actions,
                "rewards": np.pad(rewards, ((0, pad))),
                "terminations": np.pad(terminations, ((0, pad))),
                "truncations": np.pad(truncations, ((0, pad))),
            }
            if "seed" in episode_data:
                episode_batch["seed"] = np.full(len(observations), episode_data["seed"], dtype=np.uint64)
            if episode_data.get("infos", {}):
                episode_batch["infos"] = _encode_info(episode_data["infos"])
            episode_batch = pa.RecordBatch.from_pydict(episode_batch)

            total_steps += len(rewards)
            ds.write_dataset(
                episode_batch,
                self.data_path,
                format="parquet",
                partitioning=["episode_id"],
                existing_data_behavior="overwrite_or_ignore"
            )

        self.update_metadata({
            "total_steps": total_steps,
            "total_episodes": total_episodes
        })

    def update_from_storage(self, storage: MinariStorage):
        for episode in storage.get_episodes(range(storage.total_episodes)):
            del episode["id"]
            self.update_episodes([episode])
        
        authors = {self.metadata["author"], storage.metadata.get("author")}
        emails = {self.metadata["author_email"], storage.metadata.get("author_email")}
        self.update_metadata(
            {
                "author": "; ".join([aut for aut in authors if aut is not None]),
                "author_email": "; ".join([e for e in emails if e is not None]),
            }
        )


def _encode_space(space: gym.Space, values: Any, pad: int = 0):
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(values, dict)
        arrays, names = [], []
        for key, value in values.items():
            names.append(key)
            arrays.append(_encode_space(space[key], value, pad=pad))
        return pa.StructArray.from_arrays(arrays, names=names)
    if isinstance(space, gym.spaces.Tuple):
        assert isinstance(values, tuple)
        arrays, names = [], []
        for i, value in enumerate(values):
            names.append(str(i))
            arrays.append(_encode_space(space[i], value, pad=pad))
        return pa.StructArray.from_arrays(arrays, names=names)
    elif isinstance(space, gym.spaces.Box):
        values = np.asarray(values).reshape(len(values), -1)
        values = np.pad(values, ((0, pad), (0, 0)))
        dtype = pa.list_(pa.from_numpy_dtype(space.dtype), list_size=values.shape[1])
        return pa.FixedSizeListArray.from_arrays(values.reshape(-1), type=dtype)
    elif isinstance(space, gym.spaces.Discrete):
        values = np.asarray(values).reshape(len(values), -1)
        values = np.pad(values, ((0, pad), (0, 0)))
        return pa.array(values.squeeze(-1), type=pa.int32())
    elif isinstance(space, gym.spaces.Text):
        if not isinstance(values, list):
            values = list(values)
        values.extend([None] * pad)
        return pa.array(values, type=pa.string())
    else:
        raise ValueError(f"{space} is not a supported space type")
                    


def _decode_space(space, values: pa.Array):
    if isinstance(space, gym.spaces.Dict):
        return {
            name: _decode_space(subspace, values.field(name))
            for name, subspace in space.spaces.items()
        }
    elif isinstance(space, gym.spaces.Tuple):
        return tuple([
            _decode_space(subspace, values.field(str(i)))
            for i, subspace in enumerate(space.spaces)
        ])
    elif isinstance(space, gym.spaces.Box):
        data = np.stack(values.to_numpy(zero_copy_only=False))
        return data.reshape(-1, *space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return values.to_numpy()
    elif isinstance(space, gym.spaces.Text):
        return values.to_pylist()
    else:
        raise ValueError(f"Not supported space type")


def _encode_info(values: Any):
    if isinstance(values, (dict, tuple)):
        arrays, names = [], []
        iterator = values.items() if isinstance(values, dict) else enumerate(values)
        for key, value in iterator:
            data = _encode_info(value)
            arrays.append(data)
            names.append(str(key))
        return pa.StructArray.from_arrays(arrays, names=names)
    elif isinstance(values, np.ndarray) or (isinstance(values, Sequence) and isinstance(values[0], np.ndarray)):
        if isinstance(values, Sequence):
            values = np.stack(values)
        values = values.reshape(len(values), -1)
        dtype = pa.from_numpy_dtype(values.dtype)
        struct = pa.list_(dtype, list_size=values.shape[1])
        return pa.FixedSizeListArray.from_arrays(values.reshape(-1), type=struct)            
    else:
        return pa.array(list(values))


def _decode_info(values: pa.Array):
    nested_dict = {}
    for i, field in enumerate(values.type):
        if isinstance(field, pa.StructArray):
            nested_dict[field.name] = _decode_info(values.field(i))
        else:
            nested_dict[field.name] = values.field(i).to_numpy(zero_copy_only=False)
    return nested_dict