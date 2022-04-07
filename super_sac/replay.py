import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import random
import warnings
from collections import deque
from typing import Dict, Tuple, List


class ReplayBufferStorage:
    def __init__(self, size, state_example, act_example):
        self.action_stack = np.zeros((size,) + act_example.shape, dtype=np.float32)
        self.reward_stack = np.zeros((size, 1), dtype=np.float32)
        self.done_stack = np.zeros((size, 1), dtype=np.uint8)

        self.s_stack = {}
        self.s1_stack = {}
        self.s_dtypes = {}
        for label, array in state_example.items():
            shape = (size,) + array.shape
            self.s_dtypes[label] = array.dtype
            self.s_stack[label] = np.zeros(shape, dtype=array.dtype)
            self.s1_stack[label] = np.zeros(shape, dtype=array.dtype)

        self.size = size
        self._next_idx = 0
        self._max_filled = 0

    def __len__(self):
        return self._max_filled

    def add(self, s, a, r, s1, d):
        # this buffer supports batched experience
        if len(a.shape) > 1:
            # there must be a batch dimension
            num_samples = len(a)
        else:
            num_samples = 1
            r, d = np.array(r), np.array(d)

        a = a.astype(np.float32)
        r = r.astype(np.float32)
        d = d.astype(np.uint8)

        s = self._make_dict_dtype(s)
        s1 = self._make_dict_dtype(s1)

        R = np.arange(self._next_idx, self._next_idx + num_samples) % self.size
        for label in s.keys():
            self.s_stack[label][R] = s[label]
        for label in s1.keys():
            self.s1_stack[label][R] = s1[label]
        self.action_stack[R] = a
        self.reward_stack[R] = r
        self.done_stack[R] = d
        # Advance index.
        self._max_filled = min(
            max(self._next_idx + num_samples, self._max_filled), self.size
        )
        self._next_idx = (self._next_idx + num_samples) % self.size
        return R

    def _make_dict_dtype(self, dict_):
        return {key: val.astype(self.s_dtypes[key]) for key, val in dict_.items()}

    def __getitem__(self, indices):
        try:
            iter(indices)
        except ValueError:
            raise IndexError(
                "DictBufferStorage getitem called with indices object that is not iterable"
            )
        state = {}
        next_state = {}

        for label in self.s_stack.keys():
            state[label] = torch.from_numpy(self.s_stack[label][indices])
            next_state[label] = torch.from_numpy(self.s1_stack[label][indices])
        action = torch.from_numpy(self.action_stack[indices])
        if action.dim() < 2:
            action = action.unsqueeze(1)
        reward = torch.from_numpy(self.reward_stack[indices])
        done = torch.from_numpy(self.done_stack[indices])
        return (state, action, reward, next_state, done)

    def get_all_transitions(self):
        return (
            self.s_stack[: self._max_filled],
            self.action_stack[: self._max_filled],
            self.reward_stack[: self._max_filled],
            self.s1_stack[: self._max_filled],
            self.done_stack[: self._max_filled],
        )


class _BasicReplayBuffer:
    def __init__(self, size):
        self._maxsize = size
        self._storage = None

    def __len__(self):
        return len(self._storage) if self._storage is not None else 0

    def push(self, state, action, reward, next_state, done):
        if self._storage is None:
            if len(action.shape) > 1:
                act_example = action[0]
                state_example = {x: y[0] for x, y in state.items()}
            else:
                act_example = action
                state_example = state
            self._storage = ReplayBufferStorage(
                self._maxsize,
                state_example=state_example,
                act_example=act_example,
            )
        return self._storage.add(state, action, reward, next_state, done)

    def sample(self, batch_size, get_idxs=False):
        random_idxs = torch.randint(len(self._storage), (batch_size,))
        if get_idxs:
            return self._storage[random_idxs], random_idxs.numpy()
        else:
            return self._storage[random_idxs]

    def get_all_transitions(self):
        return self._storage.get_all_transitions()

    def load_experience(self, s, a, r, s1, d):
        assert len(s) <= self._maxsize, "Experience dataset is larger than the buffer."
        if len(r.shape) < 2:
            r = np.expand_dims(r, 1)
        if len(d.shape) < 2:
            d = np.expand_dims(d, 1)
        self.push(s, a, r, s1, d)


class ReplayBuffer(_BasicReplayBuffer):
    def __init__(self, size, alpha=0.6, beta=1.0):
        super(ReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self.alpha = alpha
        self.beta = beta

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.total_sample_calls = 0

    def push(self, state, action, reward, next_state, done, priorities=None, **kwargs):
        R = super().push(state, action, reward, next_state, done)
        if priorities is None:
            priorities = self._max_priority
        self._it_sum[R] = priorities ** self.alpha
        self._it_min[R] = priorities ** self.alpha

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size):
        self.total_sample_calls += 1
        idxes = self._sample_proportional(batch_size)
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-self.beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-self.beta) / max_weight
        return self._storage[idxes], torch.from_numpy(weights), idxes

    def sample_uniform(self, batch_size):
        self.total_sample_calls += 1
        return super().sample(batch_size, get_idxs=True)

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self._storage)
        self._it_sum[idxes] = priorities ** self.alpha
        self._it_min[idxes] = priorities ** self.alpha
        self._max_priority = max(self._max_priority, np.max(priorities))


def unique(sorted_array):
    """
    More efficient implementation of np.unique for sorted arrays
    :param sorted_array: (np.ndarray)
    :return:(np.ndarray) sorted_array without duplicate elements
    """
    if len(sorted_array) == 1:
        return sorted_array
    left = sorted_array[:-1]
    right = sorted_array[1:]
    uniques = np.append(right != left, True)
    return sorted_array[uniques]


class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array that supports Index arrays, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        :param capacity: (int) Total size of the array - must be a power of two.
        :param operation: (lambda (Any, Any): Any) operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: (Any) neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences
        :return: (Any) result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # indexes of the leaf
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int):
            idxs = np.array([idxs])
        # go up one level in the tree and remove duplicate indexes
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            # as long as there are non-zero indexes, update the corresponding values
            self._value[idxs] = self._operation(
                self._value[2 * idxs], self._value[2 * idxs + 1]
            )
            # go up one level in the tree and remove duplicate indexes
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=np.add, neutral_element=0.0
        )
        self._value = np.array(self._value)

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]
        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of SumSegmentTree
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum for each entry in prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        :param prefixsum: (np.ndarray) float upper bounds on the sum of array prefix
        :return: (np.ndarray) highest indexes satisfying the prefixsum constraint
        """
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1e-5
        assert isinstance(prefixsum[0], float)

        idx = np.ones(len(prefixsum), dtype=int)
        cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):  # while not all nodes are leafs
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(
                self._value[idx] <= prefixsum, prefixsum - self._value[idx], prefixsum
            )
            # prepare update of prefixsum for all right children
            idx = np.where(
                np.logical_or(self._value[idx] > prefixsum, np.logical_not(cont)),
                idx,
                idx + 1,
            )
            # Select child node for non-leaf nodes
            prefixsum = prefixsum_new
            # update prefixsum
            cont = idx < self._capacity
            # collect leafs
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=np.minimum, neutral_element=float("inf")
        )
        self._value = np.array(self._value)

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])
        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of MinSegmentTree
        """
        return super(MinSegmentTree, self).reduce(start, end)


class _Trajectory:
    """
    Trajectories build a sequence of (s, a, r, d) tuples and
    then "archive" them in the form of a zarr group that can
    be stored in memory or on disk.
    """

    def __init__(self, parallel_envs: int = 1):
        self._data = {}
        self.archived = False
        self._partial_traj = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        self.state_keys = []
        self._length = None
        self.parallel_envs = parallel_envs

    def _pad_if_not_parallel(self, s, a, r, s1, d):
        if self.parallel_envs == 1:
            s = {k: np.expand_dims(v, 0) for k, v in s.items()}
            s1 = {k: np.expand_dims(v, 0) for k, v in s1.items()}
            a = np.expand_dims(a, 0)
            while len(r.shape) < 2:
                r = np.expand_dims(r, 0)
            while len(d.shape) < 2:
                d = np.expand_dims(d, 0)
        return s, a, r, s1, d

    def push(
        self,
        s: Dict[str, np.ndarray],
        a: np.ndarray,
        r: np.ndarray,
        s1: Dict[str, np.ndarray],
        d: np.ndarray,
    ):
        # check batch shapes on the way in
        s, a, r, s1, d = self._pad_if_not_parallel(s, a, r, s1, d)
        assert isinstance(s, dict)
        for k, v in s.items():
            assert v.shape[0] == self.parallel_envs
        for k, v in s1.items():
            assert v.shape[0] == self.parallel_envs
        assert isinstance(a, np.ndarray)
        assert a.shape[0] == self.parallel_envs
        assert isinstance(r, np.ndarray)
        assert r.shape[0] == self.parallel_envs
        assert isinstance(d, np.ndarray)
        assert d.shape[0] == self.parallel_envs

        self._partial_traj["states"].append(s)
        self._partial_traj["actions"].append(a)
        self._partial_traj["rewards"].append(r)
        self._partial_traj["next_states"].append(s1)
        self._partial_traj["dones"].append(d)

    def __len__(self):
        if self.archived:
            return self._length
        return None

    def archive(self):
        assert not self.archived, "Trajectory has already been archived..."
        merged_traj = self._merge_traj()
        assert merged_traj["actions"].shape[0] == self.parallel_envs
        traj_len = merged_traj["actions"].shape[1]
        for k, v in merged_traj["states"].items():
            assert v.shape[:2] == (self.parallel_envs, traj_len)
        for k, v in merged_traj["next_states"].items():
            assert v.shape[:2] == (self.parallel_envs, traj_len)
        assert merged_traj["rewards"].shape[:2] == (self.parallel_envs, traj_len)
        assert merged_traj["dones"].shape[:2] == (self.parallel_envs, traj_len)
        assert merged_traj["states"].keys() == merged_traj["next_states"].keys()
        self._length = traj_len
        self.state_keys = merged_traj["states"].keys()
        self._data = merged_traj
        del self._partial_traj
        self.archived = True

    def __getitem__(self, idxs):
        traj_idx, slice_ = idxs
        assert traj_idx < self.parallel_envs
        assert self.archived, "Call `archive()` before slicing a Trajectory"
        assert isinstance(slice_, slice)
        assert (
            slice_.stop <= self._length
        ), f"Trajectory slice out of range for end point {slice_.stop} with length {self._length}"

        # end_idx = range(slice_.start, slice_.stop, slice_.step or 1)[-1]
        a = self._data["actions"][traj_idx, slice_]
        r = self._data["rewards"][traj_idx, slice_]
        d = self._data["dones"][traj_idx, slice_]

        s = {k: self._data["states"][k][traj_idx, slice_] for k in self.state_keys}
        s1 = {
            k: self._data["next_states"][k][traj_idx, slice_] for k in self.state_keys
        }
        return s, a, r, s1, d

    def _merge_traj(self):
        states = self._partial_traj["states"]
        actions = self._partial_traj["actions"]
        rewards = self._partial_traj["rewards"]
        next_states = self._partial_traj["next_states"]
        dones = self._partial_traj["dones"]

        merged_s = {k: [] for k in states[0].keys()}
        merged_a = []
        merged_r = []
        merged_s1 = {k: [] for k in next_states[0].keys()}
        merged_d = []
        for i, (s, a, r, s1, d) in enumerate(
            zip(states, actions, rewards, next_states, dones)
        ):
            for k, v in s.items():
                try:
                    merged_s[k].append(v)
                except KeyError:
                    raise KeyError(
                        f"Trajectory: inconsistent state keys! Key `{k}` at timestep {i} "
                        f"not found in timestep 0 with keys {merged_s.keys()}"
                    )
            for k, v in s1.items():
                try:
                    merged_s1[k].append(v)
                except KeyError:
                    raise KeyError(
                        f"Trajectory: inconsistent state keys! Key `{k}` at timestep {i} "
                        f"not found in timestep 0 with keys {merged_s1.keys()}"
                    )
            merged_a.append(a)
            merged_r.append(r)
            merged_d.append(d)

        s = {
            k: np.array(v, dtype=v[0].dtype).swapaxes(0, 1) for k, v in merged_s.items()
        }
        a = np.array(merged_a, dtype=merged_a[0].dtype).swapaxes(0, 1)
        r = np.array(merged_r, dtype=np.float32).swapaxes(0, 1)
        s1 = {
            k: np.array(v, dtype=v[0].dtype).swapaxes(0, 1)
            for k, v in merged_s1.items()
        }
        d = np.array(merged_d, dtype=bool).swapaxes(0, 1)
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "next_states": s1,
            "dones": d,
        }


class _TrajectoryBasedDataset(IterableDataset):
    """
    Build batches of transitions by slicing sequences from
    a collection of Trajectory objects. Uses the Pytorch
    IterableDataset format for multiprocessing (load the
    next batch while we're training on the previous one).
    """

    def __init__(self, trajectory_buffer, max_trajectories=1000, seq_length=10):
        self.max_trajectories = max_trajectories
        self.seq_length = seq_length
        self.trajectories = trajectory_buffer

    def _id(self):
        try:
            id_ = torch.utils.data.get_worker_info().id
        except:
            id_ = 0
        return id_

    def _extend(self, arr):
        longer = np.ones((self.seq_length,) + arr.shape[1:], dtype=arr.dtype) * arr[0]
        # longer = np.zeros((self.seq_length,) + arr.shape[1:], dtype=arr.dtype)
        longer[-len(arr) :] = arr
        return longer

    def _pad(self, s, a, r, s1, d):
        for k in s.keys():
            s[k] = self._extend(s[k])
        for k in s1.keys():
            s1[k] = self._extend(s1[k])
        a = self._extend(a)
        r = self._extend(r)
        d = self._extend(d)
        return s, a, r, s1, d

    def sample(self):
        # print(f"{self._id()} : {len(self.trajectories)}")
        traj = random.choice(self.trajectories)
        actor_idx = random.randint(0, traj.parallel_envs - 1)
        start_idx = random.randint(0, max(len(traj) - self.seq_length, 0))
        end_idx = min(start_idx + self.seq_length, len(traj))
        s, a, r, s1, d = traj[actor_idx, start_idx:end_idx]
        if len(d) < self.seq_length:
            s, a, r, s1, d = self._pad(s, a, r, s1, d)
        return s, a, r, s1, d

    def __iter__(self):
        while True:
            yield self.sample()


class TrajectoryBuffer:
    """
    Hide the Trajectory-based format and PyTorch Dataset API behind a
    standard Q-Learning interface.
    """

    def __init__(
        self,
        max_trajectories: int,
        seq_length: int,
        parallel_rollouts: int = 1,
        workers: int = 8,
    ):
        self._trajectory_buffer = deque([], maxlen=max_trajectories)
        self._dset = _TrajectoryBasedDataset(
            self._trajectory_buffer,
            max_trajectories=max_trajectories,
            seq_length=seq_length,
        )
        self.seq_length = seq_length
        self._parallel_rollouts = parallel_rollouts
        self._cur_trajectory = _Trajectory(parallel_envs=parallel_rollouts)
        self._workers = workers
        self._loader = None
        self._batch_size = None
        self._force_remake = False
        self.total_sample_calls = 0

    def __len__(self):
        return sum([len(traj) * traj.parallel_envs for traj in self._trajectory_buffer])

    def push(self, state, action, reward, next_state, done, terminate_traj=None):
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        if not isinstance(reward, np.ndarray):
            reward = np.array([reward])
        if not isinstance(done, np.ndarray):
            done = np.array([done])
        self._cur_trajectory.push(state, action, reward, next_state, done)

        if terminate_traj is None:
            termiante_traj = done.any()
        if terminate_traj:
            self._cur_trajectory.archive()
            self._trajectory_buffer.append(self._cur_trajectory)
            self._force_remake = True
            self._cur_trajectory = _Trajectory(parallel_envs=self._parallel_rollouts)

    def sample(self, batch_size):
        self.total_sample_calls += 1
        if self._loader is None or batch_size != self._batch_size or self._force_remake:
            # The force_remake thing is a hack to get new episodes into worker threads.
            # I tried but failed to figure out a more complicated multiprocessing solution.
            # This would get very inefficient if the trajectory length was short.
            if self._batch_size is not None and batch_size != self._batch_size:
                warnings.warn(
                    "Warning: changing the batch_size of a TrajectoryBuffer is not efficient and should be avoided where possible."
                )
            self._batch_size = batch_size
            self._loader = iter(
                DataLoader(
                    self._dset,
                    batch_size=self._batch_size,
                    num_workers=self._workers,
                    pin_memory=True,
                )
            )
            self._force_remake = False

        s, a, r, s1, d = next(self._loader)

        if self.seq_length == 1:
            s = {k: v.squeeze(1) for k, v in s.items()}
            s1 = {k: v.squeeze(1) for k, v in s1.items()}
            a = a.squeeze(1)
            r = r.squeeze(1)
            d = d.squeeze(1)
        return (s, a, r, s1, d), None, None

    def sample_uniform(self, batch_size):
        return self.sample(batch_size)[:-1]

    def update_priorities(self, *args, **kwargs):
        pass
