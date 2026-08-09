# Copied from https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import heapq


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:

        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seqlen_list
            ), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seqlen_list
            ), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced, this is
        used in balacing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def build_fsdp_modality_aligned_order(
    modality_flags: list[bool],
    dp_size: int,
    global_batch_size: int,
) -> list[int]:
    """Reorder mixed VLM/text samples into FSDP-safe global batches.

    FSDP data-parallel ranks must enter the same model submodules in the same
    order.  A global batch that contains visual inputs on only some ranks can
    therefore deadlock in a Qwen-VL visual/text forward.  The returned order
    places visual samples first in a batch and makes both modality counts
    multiples of ``dp_size``.  ``-1`` denotes a zero-loss visual dummy and
    ``-2`` denotes a zero-loss text dummy; non-negative values are original
    sample indices.

    The helper intentionally preserves every original sample.  At most the
    dummies needed to complete modality-aligned global batches are added.
    """
    if (
        dp_size <= 1
        or global_batch_size <= 0
        or global_batch_size % dp_size != 0
        or not modality_flags
        or all(modality_flags)
        or not any(modality_flags)
    ):
        return list(range(len(modality_flags)))

    visual_indices = [index for index, is_visual in enumerate(modality_flags) if is_visual]
    text_indices = [index for index, is_visual in enumerate(modality_flags) if not is_visual]
    result: list[int] = []
    visual_cursor = 0
    text_cursor = 0

    while visual_cursor < len(visual_indices) or text_cursor < len(text_indices):
        remaining_visual = len(visual_indices) - visual_cursor
        real_visual_count = min(remaining_visual, global_batch_size)
        aligned_visual_count = (
            ((real_visual_count + dp_size - 1) // dp_size) * dp_size if real_visual_count else 0
        )
        aligned_visual_count = min(global_batch_size, aligned_visual_count)

        result.extend(visual_indices[visual_cursor : visual_cursor + real_visual_count])
        visual_cursor += real_visual_count
        result.extend([-1] * (aligned_visual_count - real_visual_count))

        text_slots = global_batch_size - aligned_visual_count
        real_text_count = min(text_slots, len(text_indices) - text_cursor)
        result.extend(text_indices[text_cursor : text_cursor + real_text_count])
        text_cursor += real_text_count
        result.extend([-2] * (text_slots - real_text_count))

    return result


def get_fsdp_modality_aligned_partitions(
    num_samples: int,
    dp_size: int,
    global_batch_size: int,
) -> list[list[int]]:
    """Split modality-aligned global batches with identical per-rank order."""
    if dp_size <= 0 or global_batch_size <= 0 or global_batch_size % dp_size != 0:
        raise ValueError(
            f"FSDP modality alignment requires positive sizes with global_batch_size divisible by dp_size; "
            f"got num_samples={num_samples}, dp_size={dp_size}, global_batch_size={global_batch_size}"
        )
    if num_samples % global_batch_size != 0:
        raise ValueError(
            f"FSDP modality-aligned samples must fill complete global batches; "
            f"got num_samples={num_samples}, global_batch_size={global_batch_size}"
        )

    partitions = [[] for _ in range(dp_size)]
    for batch_start in range(0, num_samples, global_batch_size):
        for rank in range(dp_size):
            partitions[rank].extend(range(batch_start + rank, batch_start + global_batch_size, dp_size))
    return partitions


def get_reverse_idx(idx_map):
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map
