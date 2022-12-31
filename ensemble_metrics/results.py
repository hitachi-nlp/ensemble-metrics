import logging
from typing import Dict, List, Union, Optional, Any, Tuple

import numpy as np
from pydantic import BaseModel
from machine_learning.typing import FloatOptional

logger = logging.getLogger(__name__)


class MTIBounds(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    error_upper_bound: FloatOptional[Union[np.ndarray, float]] = None
    error_lower_bound: FloatOptional[Union[np.ndarray, float]] = None
    error_lower_bound_0: FloatOptional[Union[np.ndarray, float]] = None
    error_lower_bound_1: FloatOptional[Union[np.ndarray, float]] = None
    error_lower_bound_2: FloatOptional[Union[np.ndarray, float]] = None
    error_lower_bound_3: FloatOptional[Union[np.ndarray, float]] = None
    
    H_Y: FloatOptional[Union[np.ndarray, float]] = None
    H_Y_X: FloatOptional[Union[np.ndarray, float]] = None
    H_Y_Yhat: FloatOptional[Union[np.ndarray, float]] = None
    # H_Y_Xomega_list: FloatOptional[List[Union[np.ndarray, float]]] = None

    H_Y_Xomega_2_top3: FloatOptional[List[Union[np.ndarray, float]]] = None
    H_omega_2_variables_top3: FloatOptional[List[Union[np.ndarray, Tuple[str, ...]]]] = None
    H_Y_Xomega_2_bottom3: FloatOptional[List[Union[np.ndarray, float]]] = None
    H_omega_2_variables_bottom3: FloatOptional[List[Union[np.ndarray, Tuple[str, ...]]]] = None

    H_Y_Xomega_3_top3: FloatOptional[List[Union[np.ndarray, float]]] = None
    H_omega_3_variables_top3: FloatOptional[List[Union[np.ndarray, Tuple[str, ...]]]] = None
    H_Y_Xomega_3_bottom3: FloatOptional[List[Union[np.ndarray, float]]] = None
    H_omega_3_variables_bottom3: FloatOptional[List[Union[np.ndarray, Tuple[str, ...]]]] = None

    interaction_order: FloatOptional[Union[np.ndarray, float]] = None
    interaction_information: FloatOptional[Union[np.ndarray, float]] = None
    relevance: FloatOptional[Union[np.ndarray, float]] = None
    diversity: FloatOptional[Union[np.ndarray, float]] = None
    conditional_redundancy: FloatOptional[Union[np.ndarray, float]] = None
    redundancy: FloatOptional[Union[np.ndarray, float]] = None
    total_redundancy: FloatOptional[Union[np.ndarray, float]] = None
    I_term: FloatOptional[Union[np.ndarray, float]] = None
    L_term: FloatOptional[Union[np.ndarray, float]] = None


class EnsembleResults(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    valid_score: FloatOptional[Union[np.ndarray, float]]
    test_score: FloatOptional[Union[np.ndarray, float]]
    train_score: FloatOptional[Union[np.ndarray, float]]
    train_weighted_score: FloatOptional[Union[np.ndarray, float]]

    ensemble_method: FloatOptional[str] = None
    model_type: FloatOptional[str] = None
    num_models_per_type: FloatOptional[int] = None
    final_classifier_name: FloatOptional[str] = None
    regulator: FloatOptional[str] = None
    base_model_weights: FloatOptional[List] = None

    valid_to_test_n_fold: Optional[int] = None
    valid_to_test_fold: Optional[int] = None

    base_model_valid_corr_coef: FloatOptional[Union[np.ndarray, float]] = None
    base_model_test_corr_coef: FloatOptional[Union[np.ndarray, float]] = None

    # mti_interaction_order: FloatOptional[Union[np.ndarray, float]] = None
    valid_mti_bounds: FloatOptional[MTIBounds] = None
    test_mti_bounds: FloatOptional[MTIBounds] = None

    valid_mti_bounds_previous_research: FloatOptional[MTIBounds] = None
    test_mti_bounds_previous_research: FloatOptional[MTIBounds] = None

    base_model_valid_scores: FloatOptional[List[Union[np.ndarray, float]]] = None
    base_model_test_scores: FloatOptional[List[Union[np.ndarray, float]]] = None
    base_model_train_scores: FloatOptional[List[Union[np.ndarray, float]]] = None
    base_model_train_weighted_scores: FloatOptional[List[Union[np.ndarray, float]]] = None

    base_model_valid_score_average: FloatOptional[Union[np.ndarray, float]] = None
    base_model_test_score_average: FloatOptional[Union[np.ndarray, float]] = None
    base_model_train_score_average: FloatOptional[Union[np.ndarray, float]] = None
    base_model_train_weighted_score_average: FloatOptional[Union[np.ndarray, float]] = None

    base_model_valid_score_std: FloatOptional[Union[np.ndarray, float]] = None
    base_model_test_score_std: FloatOptional[Union[np.ndarray, float]] = None
    base_model_train_score_std: FloatOptional[Union[np.ndarray, float]] = None
    base_model_train_weighted_score_std: FloatOptional[Union[np.ndarray, float]] = None

    best_params: FloatOptional[Dict] = None


class HeadEnsembleResults(EnsembleResults):
    j_layer: FloatOptional[int] = None

    stack_method: FloatOptional[str] = None
    head_estimator_type: FloatOptional[str] = None
    num_heads_per_type: FloatOptional[int] = None
    layer_interval: FloatOptional[int] = None
    max_layers: FloatOptional[int] = None
    dropout_prob: FloatOptional[Union[float, np.ndarray]] = None
    dropout_from_second_model: FloatOptional[bool] = False
    pca_reduction_rate: FloatOptional[float] = None
    regulator: FloatOptional[str] = 'l1'
    no_grid_search: FloatOptional[bool] = False
    cv_average_embedding: FloatOptional[bool] = False

    heads: FloatOptional[List[Any]] = None
