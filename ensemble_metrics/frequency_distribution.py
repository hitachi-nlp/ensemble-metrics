from typing import List, Tuple, Union, Dict, Optional, Any, Iterable, Set, Sequence
import itertools
from collections import defaultdict
from pprint import pformat

Sample = Dict[str, Any]
HashableSample = Sequence[Tuple[str, Any]]
Condition = Sequence[Tuple[str, Any]]


class FrequencyDistribution:

    def __init__(self, master_samples: List[Sample], count_threshold: int = 0):
        self._master_samples: List[Sample] = master_samples
        self._num_master_samples = len(master_samples)
        self._count_threshold = count_threshold

        self._sample_indexes: List[int] = list(range(0, len(master_samples)))
        self._num_samples: int = None

        self._variable_names: List[str] = [name for name, val in next(iter(master_samples)).items()]
        self._master_variable_names = set(self._variable_names)

        # sample cache
        self._samples_cache: List[Tuple[int, Sample]] = []
        self._started_sample_caching = False
        self._finished_sample_caching = False

        # joint cache
        self._cached_joint_counts: Optional[Dict[HashableSample, float]] = None
        self._joint_to_sample_indexes_master: Dict[Tuple[str], Dict[Condition, List[int]]] = defaultdict(lambda : defaultdict(list))

        # conditional
        self._condition_on_master_samples: Optional[Condition] = None
        self._condition_to_sample_indexes_master: Dict[Condition, List[int]] = {}

    def init_from_parent(
        self,
        sample_indexes: Optional[int] = None,
        variable_names: Optional[List[str]] = None,
        condition_on_master_samples: Optional[Condition] = None,
        condition_to_sample_indexes_master = None,
        joint_to_sample_indexes_master = None
    ) -> None:
        self._check_variable_names(variable_names)

        self._sample_indexes = sample_indexes or self._sample_indexes
        self._variable_names = variable_names or self._variable_names
        self._condition_on_master_samples = condition_on_master_samples
        self._condition_to_sample_indexes_master = condition_to_sample_indexes_master if condition_to_sample_indexes_master is not None else self._condition_to_sample_indexes_master
        self._joint_to_sample_indexes_master = joint_to_sample_indexes_master if joint_to_sample_indexes_master is not None else self._joint_to_sample_indexes_master

    def _check_variable_names(self, variable_names: List[str]):
        for variable_name in variable_names:
            if variable_name not in self._master_variable_names:
                raise Exception(f'Unknown variable name: "{variable_name}"')

    def _sample_indexes(self, indexes: List[int]):
        # for variable_name in variable_names:
        #     if variable_name not in self._master_variable_names:
        #         raise Exception(f'Unknown variable name: "{variable_name}"')

        # pass for speedup
        pass

    def get_samples(self) -> Iterable[Any]:
        if self._finished_sample_caching:
            for idx, sample in self._samples_cache:
                yield idx, sample
            return

        do_cache = False
        if not self._started_sample_caching:
            self._started_sample_caching = True
            do_cache = True

        count = 0
        for idx in self._sample_indexes:
            sample = self._master_samples[idx]
            sliced_sample = self._slice_sample_by_variable(sample)
            if do_cache:  # get_samples() の衝突を防ぐため．
                self._samples_cache.append((idx, sliced_sample))
            yield idx, sliced_sample
            count += 1

        if do_cache:
            self._num_samples = count
            self._finished_sample_caching = True

    def _count_samples_once(self) -> int:
        if self._num_samples is None:
            count = 0
            for idx in self._sample_indexes:
                count += 1
            self._num_samples = count
        return self._num_samples

    @property
    def num_samples(self) -> int:
        return self._count_samples_once()

    @property
    def variable_names(self) -> List[str]:
        return self._variable_names

    def _slice_sample_by_variable(self, sample: Sample) -> Sample:
        _variable_names = self.variable_names
        return {name: sample[name] for name in _variable_names}

    def __call__(self, *args, **kwargs) -> Dict[Sample, Union[int, float]]:
        return self.joint(*args, **kwargs)

    def _hash_sample(self, sample: Union[Sample, HashableSample]) -> HashableSample:
        if isinstance(sample, dict):
            return tuple(sample.items())
        else:
            return sample

    def joint(self, return_counts=False) -> Dict[HashableSample, Union[int, float]]:
        if self._cached_joint_counts is not None:
            counts = self._cached_joint_counts
        elif tuple(self.variable_names) in self._joint_to_sample_indexes_master:
            sample_indexes = self._joint_to_sample_indexes_master[tuple(self.variable_names)]
            counts = defaultdict(float)
            for template_sample, indexes in sample_indexes.items():
                counts[self._hash_sample(template_sample)] = len(set(self._sample_indexes).intersection(indexes))
        else:
            counts = defaultdict(float)
            for _, sample in self.get_samples():
                counts[self._hash_sample(sample)] += 1
            self._cached_joint_counts = counts

        thresholded_counts = defaultdict(float)
        count_sum: float = 0
        for condition, count in counts.items():
            if count < self._count_threshold:
                continue
            thresholded_counts[condition] = count
            count_sum += count

        if return_counts:
            return thresholded_counts
        else:
            _count_sum = max(count_sum, 1.0)
            for condition in thresholded_counts:
                thresholded_counts[condition] /= _count_sum
            return thresholded_counts

    def marginal(self,
                 include_variables: Optional[List[str]] = None,
                 exclude_variables: Optional[List[str]] = None) -> 'FrequencyDistribution':
        if include_variables is not None and exclude_variables is not None:
            raise ValueError('Only one of "include_variables" or "exclude_variables" can be specified.')
        if exclude_variables is not None:
            include_variables = [key for key in self.variable_names
                                 if key not in exclude_variables]
        self._check_variable_names(include_variables)

        marginal_dist = FrequencyDistribution(self._master_samples, count_threshold=self._count_threshold)
        marginal_dist.init_from_parent(
            sample_indexes=self._sample_indexes,
            variable_names=[name for name in self.variable_names if name in include_variables],

            joint_to_sample_indexes_master=self._joint_to_sample_indexes_master,

            condition_on_master_samples=self._condition_on_master_samples,
            condition_to_sample_indexes_master=self._condition_to_sample_indexes_master,
        )
        return marginal_dist

    def conditional(self, condition: Union[Condition, Dict[str, Any]]) -> 'FrequencyDistribution':
        if isinstance(condition, dict):
            condition = tuple(sorted(condition.items()))
        self._check_variable_names([variable_name for variable_name, _ in condition])

        if self._condition_on_master_samples is not None:
            full_condition_on_master_samples = tuple(sorted(list(condition) + list(self._condition_on_master_samples)))
        else:
            full_condition_on_master_samples = condition

        target_sample_indexes = self._sample_indexes

        if full_condition_on_master_samples in self._condition_to_sample_indexes_master:
            target_sample_indexes = self._condition_to_sample_indexes_master[full_condition_on_master_samples]
        else:
            for single_variable_condition in full_condition_on_master_samples:
                _condition = tuple([single_variable_condition])
                if _condition in self._condition_to_sample_indexes_master:
                    that_indexes = self._condition_to_sample_indexes_master[_condition]
                    # target_sample_indexes = set(target_sample_indexes).intersection(set(that_indexes))  # SLOW
                    target_sample_indexes = _fast_intersection(iter(target_sample_indexes), iter(that_indexes))

        conditional_sample_indexes: List[int] = []
        for idx in target_sample_indexes:
            sample = self._master_samples[idx]
            if any([name in sample and sample[name] != target_val
                    for name, target_val in full_condition_on_master_samples]):
                continue
            conditional_sample_indexes.append(idx)

        condition_variable_names = [name for name, _ in full_condition_on_master_samples]
        conditional_dist = FrequencyDistribution(self._master_samples, count_threshold=self._count_threshold)
        conditional_dist.init_from_parent(
            sample_indexes=conditional_sample_indexes,
            variable_names=[name
                            for name in self.variable_names
                            if name not in condition_variable_names],

            joint_to_sample_indexes_master=self._joint_to_sample_indexes_master,

            condition_on_master_samples=full_condition_on_master_samples,
            condition_to_sample_indexes_master=self._condition_to_sample_indexes_master,
        )
        return conditional_dist

    def index_condition_sample_indexes(self, max_condition_variables: Optional[int] = None):
        max_condition_variables = max_condition_variables or len(self.variable_names) - 1
        # logger.info('indexing condition to samples ..')
        for idx, sample in self.get_samples():
            attrs = sorted(list(sample.items()))

            for num_condition_variables in range(1, max_condition_variables + 1):
                for condition_attrs in itertools.combinations(attrs, num_condition_variables):
                    cache_key_conditions = tuple(condition_attrs)
                    if cache_key_conditions in self._condition_to_sample_indexes_master:
                        self._condition_to_sample_indexes_master[cache_key_conditions].append(idx)
                    else:
                        self._condition_to_sample_indexes_master[cache_key_conditions] = [idx]
        # logger.info('indexing condition to samples finished')

    def index_joint_to_sample_indexes(self, max_joint_variables: Optional[int] = None):
        max_joint_variables = max_joint_variables or len(self.variable_names)
        for idx, sample in self.get_samples():
            attrs = sorted(list(sample.items()))

            for num_joint_variables in range(1, max_joint_variables + 1):
                for joint_attrs in itertools.combinations(attrs, num_joint_variables):
                    joint_variable_names = tuple([name for name, _ in joint_attrs])
                    sample = self._hash_sample(joint_attrs)
                    self._joint_to_sample_indexes_master[joint_variable_names][sample].append(idx)

    def __repr__(self):
        return pformat(self.joint())


def _fast_intersection(these: Iterable[int],
                       those: Iterable[int]) -> List[int]:
    """these and those must be sorted"""
    intersections = []

    try:
        this = next(these)
        that = next(those)
    except StopIteration:
        return []

    do_pop_these = True
    while True:
        if this == that:
            intersections.append(this)
        elif this < that:
            do_pop_these = True
        else:
            do_pop_these = False

        if do_pop_these:
            try:
                this = next(these)
            except StopIteration:
                break
        else:
            try:
                that = next(those)
            except StopIteration:
                break

    return intersections
