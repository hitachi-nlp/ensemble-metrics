import logging
from typing import List, Optional, Tuple

import numpy as np

from ensemble_metrics.stats import (
    FrequencyDistribution,
    calc_entropy,
    calc_conditional_entropy,
    calc_multi_information,
    calc_conditional_multi_information,
    safe_log2
)
from ensemble_metrics.results import MTIBounds, EnsembleResults, HeadEnsembleResults

logger = logging.getLogger(__name__)


def calc_stats(
    # ensemble_model: BaseEstimator,
    ensemble_preds: np.ndarray,

    # base_models: List[BaseEstimator],
    preds_of_base_models: List[np.ndarray],

    # indexes: np.ndarray,

    y,

    interaction_order: Optional[int] = None,
    # original_base_models: Optional[List[BaseEstimator]] = None,
    concavity_p0: Optional[float] = None,
) -> Tuple[MTIBounds, MTIBounds]:

    # global _MTI_BASE_MODEL_DEPENDENT_QUANTITY_CACHE
    # cache_key = tuple([
    #     id(base_models) if original_base_models is None else id(original_base_models),
    #     id(indexes),
    #     id(y),
    #     interaction_order,
    # ])
    # cache = _MTI_BASE_MODEL_DEPENDENT_QUANTITY_CACHE[cache_key]

    cache = {}  # XXX fake. will not be updated

    if len(cache) > 0:
        logger.info('use mti cache')
    # print(cache_key)
    # print('cache:', pformat(cache))

    def calc_error_lower_bound(conditional_entropy: float,  # H(Y|X)
                               num_unique_labels: int,
                               estimation_type: str,
                               I_term: float = None,
                               L_term: float = None,
                               concavity_m: float = 4.0,
                               concavity_p0: Optional[float] = None) -> float:
        uncertainty_terms = conditional_entropy
        if I_term is not None:
            uncertainty_terms += I_term
        if L_term is not None:
            uncertainty_terms += L_term

        if estimation_type == 'previous_research':
            return (uncertainty_terms - 1) / safe_log2(num_unique_labels)
        else:
            delta = 0.01
            bins = int(0.5 / delta)

            if estimation_type == 'bug':
                lower_bound = 1.0
            else:
                lower_bound = -99999

            for i_p0 in range(0, bins - 1):
                p0 = delta * (i_p0 + 1)  # 0は除外する．
                if concavity_p0 is not None:
                    p0 = concavity_p0

                if estimation_type == 'strong_concavity':
                    H_p0 = - p0 * safe_log2(p0) - (1 - p0) * safe_log2(1 - p0)
                    H_dash_p0 = - safe_log2(p0) + safe_log2(1 - p0)
                    U_p0 = H_p0 + p0 * safe_log2(num_unique_labels - 1)
                    U_dash_p0 = H_dash_p0 + safe_log2(num_unique_labels - 1)

                    if concavity_m == 0.1:   # 発散を防ぐため
                        bound = p0 + (uncertainty_terms - U_p0) / U_dash_p0
                    else:
                        bound = p0\
                            + (U_dash_p0 / concavity_m)\
                            * (1 - np.sqrt(1 - 2 * concavity_m / np.square(U_dash_p0) * (uncertainty_terms - U_p0)))
                elif estimation_type == 'bug':
                    H_p0 = p0 * safe_log2(p0) + (1 - p0) * safe_log2(1 - p0)
                    H_dash_p0 = safe_log2(p0 / (1 - p0))
                    bound = (p0 + (uncertainty_terms - H_p0) / H_dash_p0) / (1 + safe_log2(num_unique_labels - 1) / H_dash_p0)
                else:
                    raise NotImplementedError()

                if estimation_type == 'bug':
                    if bound < lower_bound:
                        lower_bound = bound
                else:
                    if bound > lower_bound:
                        lower_bound = bound

                if concavity_p0 is not None:
                    logger.info(f'concavity_p0: {str(concavity_p0)}')
                    break

        return lower_bound

    # Build Distribution
    # ensemble_preds = ensemble_model.predict(indexes)
    # preds_of_base_models = []
    # for model in base_models:
    #     preds = model.predict(indexes)
    #     preds_of_base_models.append(preds)

    # labels = [y[idx] for idx in indexes]
    labels = y

    Es = [1 if label != ensemble_preds[i] else 0
          for i, label in enumerate(labels)]

    samples = []
    feature_names = [f'pred_{i_model}' for i_model in range(0, len(base_models))]
    for i_sample, (ensemble_pred, E, label) in enumerate(zip(ensemble_preds, Es, labels)):
        features = []
        for i_model, feature_name in enumerate(feature_names):
            pred = preds_of_base_models[i_model][i_sample]
            features.append((feature_name, pred))
        features.append(('label', label))
        features.append(('pred_ensemble', ensemble_pred))
        features.append(('E', E))
        # samples.append(features)
        samples.append(dict(features))

    dist = FrequencyDistribution(
        samples,
        count_threshold = 0 if interaction_order is None else 3
    )

    # Calculate Statistics
    if 'relevances' not in cache:
        relevances = []
        for feature_name in feature_names:
            relevances.append(
                calc_multi_information(
                    dist.marginal(include_variables=[feature_name, 'label']),
                    interaction_order=interaction_order
                )
            )
        cache['relevances'] = relevances
    relevances = cache['relevances']
    relevance = np.sum(relevances)
    logger.info('{0:<30}{1:<.3f}'.format('relevance', relevance))

    def calc_redundancy(type_: str, interaction_order: int):
        if type_ == 'redundancy':
            return calc_multi_information(
                dist.marginal(exclude_variables=['label', 'pred_ensemble', 'E']),
                interaction_order=interaction_order
            )
        elif type_ == 'conditional_redundancy':
            return calc_conditional_multi_information(
                dist.marginal(exclude_variables=['pred_ensemble', 'E']),
                ['label'],
                interaction_order=interaction_order,
            )
        elif type_ == 'diff_redundancy':
            return calc_multi_information(
                dist.marginal(exclude_variables=['pred_ensemble', 'E']),
                interaction_order=interaction_order,
                condition_variables_Z=['label'],
            )
        else:
            raise ValueError(f'Unknown type={type_}')

    if 'redundancy' not in cache:
        cache['redundancy'] = calc_redundancy('redundancy', interaction_order)
    redundancy = cache['redundancy']
    logger.info('{0:<30}{1:<.3f}'.format('redundancy:', redundancy))

    if 'conditional_redundancy' not in cache:
        cache['conditional_redundancy'] = calc_redundancy('conditional_redundancy',
                                                          interaction_order)
    conditional_redundancy = cache['conditional_redundancy']
    logger.info('{0:<30}{1:<.3f}'.format('conditional_redundancy:',
                                         conditional_redundancy))

    if 'diff_redundancy' not in cache:
        cache['diff_redundancy'] = calc_redundancy('diff_redundancy', interaction_order)
    diff_redundancy = cache['diff_redundancy']
    logger.info('{0:<30}{1:<.3f}'.format('diff_redundancy:', diff_redundancy))
    total_redundancy = diff_redundancy

    diversity = - total_redundancy
    interaction_information_I_X_Y = relevance + diversity  # I(X;Y)

    if 'label_entropy_H_Y' not in cache:
        cache['label_entropy_H_Y'] = calc_entropy(dist.marginal(include_variables=['label']))  # H(Y)
    label_entropy_H_Y = cache['label_entropy_H_Y']
    logger.info('{0:<30}{1:<.3f}'.format('label_entropy_H_Y:', label_entropy_H_Y))

    H_Y_X = label_entropy_H_Y - interaction_information_I_X_Y  # H(Y|X)
    logger.info('{0:<30}{1:<.3f}'.format('H_Y_X:', H_Y_X))

    H_Y_Yhat = calc_conditional_entropy(
        dist.marginal(include_variables=['pred_ensemble', 'label']),
        ['pred_ensemble'],
    )
    logger.info('{0:<30}{1:<.3f}'.format('H_Y_Yhat:', H_Y_Yhat))

    def _get_omega_statistics(order: int):
        marginal = dist.marginal(exclude_variables=['pred_ensemble', 'E'])
        vals, variables = calc_conditional_entropy(
            marginal,
            [name for name in marginal.variable_names if name.startswith('pred_')],
            interaction_order=order,
            return_all_omega=True,
        )
        sorted_vals = np.sort(vals).tolist()
        sorted_variables = [variables[i] for i in np.argsort(vals)]
        return sorted_vals[:3], sorted_variables[:3], sorted_vals[-3:], sorted_variables[-3:]

    # omega_cache_key = tuple([
    #     id(base_models) if original_base_models is None else id(original_base_models),
    #     id(indexes),
    #     id(y),
    # ])
    # omega_cache = _MTI_BASE_MODEL_DEPENDENT_QUANTITY_CACHE[omega_cache_key]

    omega_cache = {}   # XXX: fake. will not be updated

    if len(omega_cache) > 0:
        logger.info('use mti omega cache')

    if 'omega_2' not in omega_cache:
        omega_cache['omega_2'] = _get_omega_statistics(2)
    H_Y_Xomega_2_top3,\
        H_omega_2_variables_top3,\
        H_Y_Xomega_2_bottom3,\
        H_omega_2_variables_bottom3 = omega_cache['omega_2']
    logger.info('{0:<30}(k={1}): {2:<.3f}  {3:<.3f}'.format('H_Y_Xomega',
                                                            2,
                                                            min(H_Y_Xomega_2_top3),
                                                            max(H_Y_Xomega_2_bottom3)))

    if 'omega_3' not in omega_cache:
        omega_cache['omega_3'] = _get_omega_statistics(3)
    H_Y_Xomega_3_top3,\
        H_omega_3_variables_top3,\
        H_Y_Xomega_3_bottom3,\
        H_omega_3_variables_bottom3 = omega_cache['omega_3']
    logger.info('{0:<30}(k={1}): {2:<.3f}  {3:<.3f}'.format('H_Y_Xomega',
                                                            3,
                                                            min(H_Y_Xomega_3_top3),
                                                            max(H_Y_Xomega_3_bottom3)))

    L_term = H_Y_Yhat - H_Y_X
    logger.info('{0:<30}{1:<.3f}'.format('L_term:', L_term))

    if 'I_term' not in cache:
        cache['I_term'] = calc_multi_information(
            dist.marginal(include_variables=['E', 'pred_ensemble']),
            interaction_order=interaction_order
        )
    I_term = cache['I_term']
    logger.info('{0:<30}{1:<.3f}'.format('I_term:', I_term))

    unique_labels = set(y)

    def calc_bounds(estimation_type: str):
        error_upper_bound = (H_Y_X) / 2.0
        error_lower_bound_1 = calc_error_lower_bound(
            H_Y_X,
            len(unique_labels),
            estimation_type,
            I_term=I_term,
            L_term=L_term,
            concavity_p0=concavity_p0,
        )
        error_lower_bound_2 = calc_error_lower_bound(
            H_Y_X,
            len(unique_labels),
            estimation_type,
            L_term=L_term,
            concavity_p0=concavity_p0,
        )
        error_lower_bound_3 = calc_error_lower_bound(
            H_Y_X,
            len(unique_labels),
            estimation_type,
            concavity_p0=concavity_p0,
        )
        return error_upper_bound, error_lower_bound_1, error_lower_bound_2, error_lower_bound_3

    our_error_upper_bound, our_error_lower_bound_1, our_error_lower_bound_2, our_error_lower_bound_3 = calc_bounds('strong_concavity')
    previous_error_upper_bound, previous_error_lower_bound_1, previous_error_lower_bound_2, previous_error_lower_bound_3 = calc_bounds('previous_research')

    our_results = MTIBounds(
        error_upper_bound=our_error_upper_bound,
        error_lower_bound=our_error_lower_bound_3,
        error_lower_bound_1=our_error_lower_bound_1,
        error_lower_bound_2=our_error_lower_bound_2,
        error_lower_bound_3=our_error_lower_bound_3,

        H_Y=label_entropy_H_Y,
        H_Y_X=H_Y_X,
        H_Y_Yhat=H_Y_Yhat,

        H_Y_Xomega_2_top3=H_Y_Xomega_2_top3,
        H_Y_Xomega_2_bottom3=H_Y_Xomega_2_bottom3,
        H_omega_2_variables_top3=H_omega_2_variables_top3,
        H_omega_2_variables_bottom3=H_omega_2_variables_bottom3,

        H_Y_Xomega_3_top3=H_Y_Xomega_3_top3,
        H_Y_Xomega_3_bottom3=H_Y_Xomega_3_bottom3,
        H_omega_3_variables_top3=H_omega_3_variables_top3,
        H_omega_3_variables_bottom3=H_omega_3_variables_bottom3,

        interaction_order=interaction_order,
        interaction_information=interaction_information_I_X_Y,
        relevance=relevance,
        diversity=diversity,
        conditional_redundancy=conditional_redundancy,
        redundancy=redundancy,
        total_redundancy=total_redundancy,
        I_term=I_term,
        L_term=L_term,
    )

    previous_results = our_results.copy()
    previous_results.error_upper_bound = previous_error_upper_bound
    previous_results.error_lower_bound = previous_error_lower_bound_3
    previous_results.error_lower_bound_1 = previous_error_lower_bound_1
    previous_results.error_lower_bound_2 = previous_error_lower_bound_2
    previous_results.error_lower_bound_3 = previous_error_lower_bound_3

    return our_results, previous_results
