import logging
from typing import List, Tuple, Union, Dict, Optional, Any, Sequence, Iterable, Set
import itertools
import math
from collections import defaultdict
from pprint import pformat

import numpy as np
from pydantic import BaseModel

from .frequency_distribution import FrequencyDistribution

logger = logging.getLogger(__name__)


class EnsembleMetrics(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    tight_LB_wo_combloss: Union[np.ndarray, float] = None
    tight_LB_w_combloss: Union[np.ndarray, float] = None
    loose_LB_wo_combloss: Union[np.ndarray, float] = None
    loose_LB_w_combloss: Union[np.ndarray, float] = None
    UB: Union[np.ndarray, float] = None

    relevance: Union[np.ndarray, float] = None
    redundancy: Union[np.ndarray, float] = None
    combloss: Union[np.ndarray, float] = None
    strength_E: Union[np.ndarray, float] = None

    # -- auxiliary quantities "omega" in the paper
    H_Y_Oomega_3_top3: List[Union[np.ndarray, float]] = None
    H_omega_3_variables_top3: List[Union[np.ndarray, Tuple[str, ...]]] = None
    H_Y_Oomega_3_bottom3: List[Union[np.ndarray, float]] = None
    H_omega_3_variables_bottom3: List[Union[np.ndarray, Tuple[str, ...]]] = None


def compute_metrics(
    labels,
    base_model_preds: List[np.ndarray],
    ensemble_preds: np.ndarray,
    p0: Optional[float] = None,  # Approximate error rate. The error rate of a base model can be used.
    MTI_k: int = 3,
) -> EnsembleMetrics:

    # -- make distribution
    samples = []
    feature_names = [f'pred_{i_model}' for i_model in range(len(base_model_preds))]
    for i_sample, (ensemble_pred, label) in enumerate(zip(ensemble_preds, labels)):
        features = []
        for i_model, feature_name in enumerate(feature_names):
            pred = base_model_preds[i_model][i_sample]
            features.append((feature_name, pred))
        features.append(('label', label))
        features.append(('pred_ensemble', ensemble_pred))
        samples.append(dict(features))
    dist = FrequencyDistribution(samples, count_threshold=0 if MTI_k is None else 3)

    # -- calculate each quantity
    relevance = np.sum(
        calc_multi_information(
            dist.marginal(include_variables=[feature_name, 'label']),
            order=MTI_k
        )
        for feature_name in feature_names
    )
    redundancy = _compute_redundancy(dist, MTI_k)

    I_O_Y = relevance - redundancy  # I(O,Y)

    H_Y = calc_entropy(dist.marginal(include_variables=['label']))  # H(Y)

    H_Y_O = H_Y - I_O_Y  # H(Y|O)
    H_Y_Yhat = calc_conditional_entropy(
        dist.marginal(include_variables=['pred_ensemble', 'label']),
        ['pred_ensemble'],
    )
    combloss = H_Y_Yhat - H_Y_O

    strength_E = I_O_Y - combloss

    # -- compute the bounds

    def compute_lower_bounds(bound_func_B_type: str) -> Tuple[float, float]:
        n_unique_labels = len(set(labels))
        LB_wo_combloss = _bound_func_B(bound_func_B_type,
                                       p0,
                                       H_Y,
                                       I_O_Y,
                                       n_unique_labels)
        LB_w_combloss = _bound_func_B(bound_func_B_type,
                                      p0,
                                      H_Y,
                                      strength_E,
                                      n_unique_labels)

        return LB_wo_combloss, LB_w_combloss

    tight_LB_wo_combloss, tight_LB_w_combloss = compute_lower_bounds('tight')
    loose_LB_wo_combloss, loose_LB_w_combloss = compute_lower_bounds('loose')
    UB = (H_Y_O) / 2.0  # from the previous paper

    # -- auxiliary quantities "omega" in the paper
    def _compute_omegas(order_n: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        marginal = dist.marginal(exclude_variables=['pred_ensemble'])
        vals, variables = calc_conditional_entropy(
            marginal,
            [name for name in marginal.variable_names if name.startswith('pred_')],
            order=order_n,
            return_all_omega=True,
        )
        sorted_vals = np.sort(vals).tolist()
        sorted_variables = [variables[i] for i in np.argsort(vals)]
        return sorted_vals[:3], sorted_variables[:3], sorted_vals[-3:], sorted_variables[-3:]

    H_Y_Oomega_3_top3,\
        H_omega_3_variables_top3,\
        H_Y_Oomega_3_bottom3,\
        H_omega_3_variables_bottom3 = _compute_omegas(3)

    return EnsembleMetrics(
        tight_LB_wo_combloss=tight_LB_wo_combloss,
        tight_LB_w_combloss=tight_LB_w_combloss,
        loose_LB_wo_combloss=loose_LB_wo_combloss,
        loose_LB_w_combloss=loose_LB_w_combloss,
        UB=UB,

        relevance=relevance,
        redundancy=redundancy,
        combloss=combloss,
        strength_E=strength_E,

        H_Y_Oomega_3_top3=H_Y_Oomega_3_top3,
        H_Y_Oomega_3_bottom3=H_Y_Oomega_3_bottom3,
        H_omega_3_variables_top3=H_omega_3_variables_top3,
        H_omega_3_variables_bottom3=H_omega_3_variables_bottom3,
    )


def _compute_redundancy(dist: FrequencyDistribution, MTI_k: int, type_='total_redundancy'):
    # see the previous paper for these decompositions of redundancy
    if type_ == 'redundancy':
        return calc_multi_information(
            dist.marginal(exclude_variables=['label', 'pred_ensemble', 'E']),
            order=MTI_k
        )
    elif type_ == 'conditional_redundancy':
        return calc_conditional_multi_information(
            dist.marginal(exclude_variables=['pred_ensemble', 'E']),
            ['label'],
            order=MTI_k,
        )
    elif type_ == 'total_redundancy':
        return calc_multi_information(
            dist.marginal(exclude_variables=['pred_ensemble', 'E']),
            order=MTI_k,
            condition_variables_Z=['label'],
        )
    else:
        raise ValueError(f'Unknown type={type_}')


def _bound_func_B(type_: str,
                  p0: float,
                  H_Y: float,
                  strength: float,
                  num_unique_labels: int,
                  concavity_m: float = 4.0) -> float:
    if type_ == 'loose':
        return (H_Y - strength - 1) / safe_log2(num_unique_labels)

    elif type_ == 'tight':
        H_p0 = - p0 * safe_log2(p0) - (1 - p0) * safe_log2(1 - p0)
        H_dash_p0 = - safe_log2(p0) + safe_log2(1 - p0)
        U_p0 = H_p0 + p0 * safe_log2(num_unique_labels - 1)
        U_dash_p0 = H_dash_p0 + safe_log2(num_unique_labels - 1)

        return p0\
            + (U_dash_p0 / concavity_m)\
            * (1 - np.sqrt(1 - 2 * concavity_m / np.square(U_dash_p0) * (H_Y - strength - U_p0)))

    else:
        raise ValueError()


def safe_log2(log_val) -> float:
    if log_val < 1e-10:
        log_val = 1e-10
    return math.log2(log_val)


def calc_multi_information(dist: FrequencyDistribution,
                           order: int = None,
                           condition_variables_Z: List[str] = None) -> float:
    """condition_variables_ZがNoneでない場合，I(X) - I(X|condition_variables_Z) を計算する．"""

    condition_variables_Z = condition_variables_Z or []
    marginal_dist = dist.marginal(exclude_variables=condition_variables_Z)

    if order is None:
        if condition_variables_Z:
            MI_marginal = _calc_multi_information_by_joint_dst(marginal_dist)
            MI_conditional = calc_conditional_multi_information(dist, condition_variables_Z)
            return MI_marginal - MI_conditional
        else:
            return _calc_multi_information_by_joint_dst(marginal_dist)
    else:
        # sum(i) {H(X_i) - H(X_i|X_(1:i-1))}

        if order is not None:
            marginal_dist.index_condition_sample_indexes(max_condition_variables=order - 1)
            marginal_dist.index_joint_to_sample_indexes(max_joint_variables=order - 1)

        if order <= 1:
            raise ValueError()
        variable_names = list(marginal_dist.variable_names)

        sum_ = 0.0
        for i_target, target_variable_name in enumerate(variable_names):
            # I(Xi|X_1:i-1) = max{ I(X_i|Omega) } を計算する．

            if i_target == 0:
                continue

            H_Xi = calc_entropy(marginal_dist.marginal(include_variables=[target_variable_name]))

            right_condition_variable_names = variable_names[:i_target]
            I_Xi_omega_list: List[float] = []
            _condition_order = min(len(right_condition_variable_names), order - 1)
            for _omega_variable_names in itertools.combinations(right_condition_variable_names,
                                                                _condition_order):
                omega_marginal_dist = marginal_dist.marginal(
                    include_variables=[target_variable_name] + list(_omega_variable_names)
                )
                H_Xi_omega = calc_conditional_entropy(  # SLOW!, たくさん呼ばれる．
                    omega_marginal_dist,
                    _omega_variable_names,
                )
                I_Xi_omega_list.append(H_Xi - H_Xi_omega)

                if condition_variables_Z:
                    H_Xi_Y = calc_conditional_entropy(
                        dist.marginal(include_variables=[target_variable_name] + condition_variables_Z),
                        condition_variables_Z,
                    )
                    omega_Z_marginal_dist = dist.marginal(
                        include_variables=[target_variable_name] + list(_omega_variable_names) + condition_variables_Z
                    )
                    H_Xi_omega_Z = calc_conditional_entropy(  # SLOW!, たくさん呼ばれる．
                        omega_Z_marginal_dist,
                        list(_omega_variable_names) + condition_variables_Z,
                    )
                    I_Xi_omega_list[-1] -= (H_Xi_Y - H_Xi_omega_Z)  # H_Xi_omega_Z - H_Xi_omega
                    # この量は必ず負になる気がする．一方で，Iの差分は正になる気がする．矛盾する．なぜ？

            max_I_Xi_omega = max(I_Xi_omega_list)
            sum_ += max_I_Xi_omega
        return sum_


def _check_freq_sum(freq_sum: float) -> float:
    threshold = 0.1
    if freq_sum < threshold:
        logger.warning('Overall frequency sum %f was smaller than %f. This will produce meaningless quantities. Possible cause is the count threshold of distributions.', freq_sum, threshold)

    non_zero_threshold = 0.000001
    if freq_sum < non_zero_threshold:
        return non_zero_threshold
    else:
        return freq_sum


def calc_conditional_multi_information(dist: FrequencyDistribution,
                                       condition_variables: List[str],
                                       order: int = None) -> float:
    prior = dist.marginal(include_variables=condition_variables)
    ret = 0.0
    freq_sum = 0.0
    for variables, freq in prior.joint().items():
        conditional = dist.conditional(dict(variables))
        if len(conditional.joint()) == 0:  # thresholdに殺された場合
            continue

        ret += freq * calc_multi_information(conditional, order=order)
        freq_sum += freq

    freq_sum = _check_freq_sum(freq_sum)
    return ret / freq_sum


def calc_entropy(dist: FrequencyDistribution) -> float:
    ret = 0.0
    freq_sum = 0.0

    for _, freq in dist.joint().items():
        ret -= freq * safe_log2(freq)
        freq_sum += freq

    freq_sum = _check_freq_sum(freq_sum)
    return ret / freq_sum


def calc_conditional_entropy(dist: FrequencyDistribution,
                             condition_variables: Sequence[str],
                             order: int = None,
                             return_all_omega: bool = False) -> Union[float, Tuple[List[float], List[Tuple[str, ...]]]]:
    if order is None:
        prior = dist.marginal(include_variables=condition_variables)
        ret = 0.0
        freq_sum = 0.0    # thresholdに殺された場合, 1.0にならない．

        for variables, freq in prior.joint().items():
            conditional = dist.conditional(dict(variables))   # SLOW
            if len(conditional.joint()) == 0:  # thresholdに殺された場合
                continue

            ret += freq * calc_entropy(conditional)
            freq_sum += freq

        freq_sum = _check_freq_sum(freq_sum)
        return ret / freq_sum
    else:
        if order < 2:
            raise ValueError()
        target_variable_names = [name for name in dist.variable_names
                                 if name not in condition_variables]
        H_Xi_omegas: List[float] = []
        _condition_order = min(len(condition_variables), order - 1)
        omega_variables: List[Tuple[str, ...]] = []
        for _omega_variable_names in itertools.combinations(condition_variables, _condition_order):
            marginal = dist.marginal(include_variables=target_variable_names + list(_omega_variable_names))
            H_Xi_omega = calc_conditional_entropy(marginal, _omega_variable_names)
            H_Xi_omegas.append(H_Xi_omega)
            omega_variables.append(_omega_variable_names)
        if return_all_omega:
            return H_Xi_omegas, omega_variables
        else:
            return min(H_Xi_omegas)


def _calc_multi_information_by_joint_dst(dist: FrequencyDistribution) -> float:
    unigram_dists = {}
    for variable_name in dist.variable_names:
        unigram_dists[variable_name] = dist.marginal(include_variables=[variable_name])

    ret = 0.0
    freq_sum = 0
    for variables, joint_freq in dist.joint().items():
        X_dict = dict(variables)

        products_of_unigrams_freqs = 1.0
        for variable_name in dist.variable_names:
            variable_val = X_dict[variable_name]
            unigram_dist = unigram_dists[variable_name]
            unigram_freq = unigram_dist.joint().get(((variable_name, variable_val),), 0.0)
            products_of_unigrams_freqs *= unigram_freq

        if products_of_unigrams_freqs != 0.0:
            ret += joint_freq\
                * safe_log2(joint_freq / products_of_unigrams_freqs)
            freq_sum += joint_freq

    freq_sum = _check_freq_sum(freq_sum)
    return ret
