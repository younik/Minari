import platform
import shutil
import tempfile

import numpy as np
from gymnasium import spaces
from minari.dataset.step_data import StepData

from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage
import timeit
import matplotlib.pyplot as plt


DATASET_SIZE = [512, 2048, 8192]
BATCH_SIZE = [8, 64, 512]
DATA_FORMATS = ["hdf5", "arrow"]
TEST_SPACES = {
    "box": spaces.Box(-1, 1, shape=(32,)),
    "discrete": spaces.Discrete(2048),
    # "text": spaces.Text(max_length=100),
    # "image": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
    # "dict": spaces.Dict({
    #     "image": spaces.Box(-1, 1, shape=(10,)),
    #     "discrete": spaces.Discrete(10),
    # }),
    # "tuple": spaces.Tuple([
    #     spaces.Box(-1, 1, shape=(10,)),
    #     spaces.Discrete(10),
    # ]),
}

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

def time_storage(space, dataset_size, batch_sizes, data_format):
    action_space = spaces.Box(-1, 1, shape=(1,))
    observation_space = space
    episodes = [
        _generate_episode_buffer(
            observation_space, action_space, length=512
        )
        for _ in range(dataset_size)
    ]

    tmp_dir = tempfile.mkdtemp()
    storage = MinariStorage.new(
        data_path=tmp_dir,
        observation_space=observation_space,
        action_space=action_space,
        data_format=data_format,
    )
    
    time_add = timeit.timeit(lambda: storage.update_episodes(episodes), number=1)
    
    time_gets = []
    for batch_size in batch_sizes:
        if batch_size <= dataset_size:
            episodes_id = np.random.randint(0, dataset_size, size=batch_size)
            time_get = timeit.timeit(lambda: storage.get_episodes(episodes_id), number=1)
            time_gets.append((batch_size, time_get))
    
    shutil.rmtree(tmp_dir)
    return time_add, time_gets

def plot_times():
    fig, ax = plt.subplots(len(TEST_SPACES), 2, figsize=(12, 25))
    width = 0.25
    x_labels = [(b, n) for b in BATCH_SIZE for n in DATASET_SIZE if b <= n]
    arange_steps = np.arange(len(x_labels))

    for plot_id, (space_name, space) in enumerate(TEST_SPACES.items()):
        for x_id, data_format in enumerate(DATA_FORMATS):
            plt_ax = ax[plot_id]
            time_adds, time_gets = [], []
            
            for n_episode in DATASET_SIZE:
                time_add, batch_times = time_storage(space, n_episode, BATCH_SIZE, data_format)                
                time_adds.append(time_add)
                    
                for batch_size, time_get in batch_times:
                    time_gets.append(time_get)
                    print(f"{space_name} n={n_episode} b={batch_size} {data_format} add: {time_add}s get: {time_get}s", flush=True)
            
            offset = width * x_id
            rects = plt_ax[0].bar(np.arange(len(DATASET_SIZE)) + offset, time_adds, width, label=data_format)
            plt_ax[0].bar_label(rects, padding=3, fmt='%.2f')
            rects = plt_ax[1].bar(arange_steps + offset, time_gets, width, label=data_format)
            plt_ax[1].bar_label(rects, padding=3, fmt='%.2f')

        plt_ax[0].set_ylabel('Time (s)')
        plt_ax[0].set_xlabel('num_episodes')
        plt_ax[0].set_title('Time write dataset with %s space' % space_name)
        plt_ax[0].set_xticks(np.arange(len(DATASET_SIZE)) + width / 2)
        plt_ax[0].set_xticklabels(DATASET_SIZE, rotation=45)
    
        plt_ax[1].set_ylabel('Time (s)')
        plt_ax[1].set_xlabel('(batch_size, num_episodes)') 
        plt_ax[1].set_title('Time read dataset with %s space' % space_name)
        plt_ax[1].set_xticks(arange_steps + width / 2)
        plt_ax[1].set_xticklabels(x_labels, rotation=45)

    fig.legend(DATA_FORMATS, loc='upper left')
    cpu_name = platform.processor()
    fig.suptitle(f'Arrow/HDF5 benchmark ({cpu_name})')
    fig.tight_layout()
    fig.savefig('time_storage.pdf')
    plt.show()

if __name__ == "__main__":
    plot_times()