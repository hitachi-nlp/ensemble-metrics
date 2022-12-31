import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any, Sequence
import json
from pprint import pformat
from collections import OrderedDict, defaultdict

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import glob

from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble._base import BaseEnsemble, _BaseHeterogeneousEnsemble
from sklearn.ensemble import StackingClassifier, VotingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from machine_learning.estimators import AutoFactory
from machine_learning.seed import Seed
from machine_learning.data_processing import statistics
from machine_learning.dataset.preprocessing.np_array import safe_log
from machine_learning.estimators.base import GridSearchConfig
from logger_setup import setup as setup_logger
from image_recognition.tasks import TaskBase
from ensemble_zoo.training_scripts import JiantTrainScript, ImageTasksTrainScript, TrainScriptConfig, TrainScriptBase
from ensemble_zoo.boosting.estimators.inference import build as build_boosting_inference, BoostingInferenceBase
from ensemble_zoo.training_scripts.base import UncompletedTrainingError
from ensemble_zoo.statistics import get_sample_scoring_function
from MoE import MoETransformerClassifier, MoETransformerClassifierConfig
from .stats import (
    FrequencyDistribution,
    calc_entropy,
    calc_conditional_entropy,
    calc_multi_information,
    calc_conditional_multi_information,
    safe_log2
)
from .results import MTIBounds, EnsembleResults, HeadEnsembleResults

logger = logging.getLogger(__name__)


class EnsembleBaseExceptoin(Exception):
    pass


class ConfigNotFoundError(EnsembleBaseExceptoin):

    def __init__(self, model_type, i_model):
        message = f'<ConfigNotFoundError(model_type={model_type}, i_model={i_model})>'
        super().__init__(message)


def _get_real_model(model, model_type = None) -> Optional[BaseEstimator]:
    if isinstance(model, Pipeline):
        if model_type == Pipeline:
            return model
        else:
            return _get_real_model(model.steps[-1][1], model_type=model_type)
    elif isinstance(model, GridSearchCV):
        if model_type == GridSearchCV:
            return model
        else:
            if hasattr(model, 'best_estimator_'):
                return _get_real_model(model.best_estimator_, model_type=model_type)
            else:
                return _get_real_model(model.estimator, model_type=model_type)
    elif model_type is None or isinstance(model, model_type):
        return model
    else:
        return None


def build_training_script(task: str,
                          image_dataset_tsv_dir: Optional[Union[str, Path]] = None,
                          images_dir: Optional[Union[str, Path]] = None,
                          log_level='INFO') -> TrainScriptConfig:
    is_image_task = task in TaskBase.list_available()
    if is_image_task:
        image_dataset_tsv_dir = f'{image_dataset_tsv_dir}/{task}/'
        images_dir = f'{images_dir}/{task}/images/'
        training_script = ImageTasksTrainScript(
            image_dataset_tsv_dir,
            images_dir,
            None,
            None,
            log_level=log_level,
        )
    else:

        training_script = JiantTrainScript(
            None,
            None,
            log_level=log_level,
        )
    return training_script


_DATASET_CACHE = {}


def load_dataset(task: str,
                 training_script: TrainScriptBase,
                 n_fold: int,
                 folds: List[int],
                 valid_to_test_n_fold: Optional[int] = None,
                 valid_to_test_fold: Optional[int] = None,
                 return_hidden_states: bool = False,
                 cv_average_embedding: bool = False,
                 model_dir: Union[str, Path] = None,
                 max_samples_per_fold: Optional[int] = None):
    cache_key = (
        task,
        id(training_script),
        n_fold,
        tuple(folds),
        valid_to_test_n_fold,
        valid_to_test_fold,
        return_hidden_states,
        cv_average_embedding,
        str(model_dir),
    )
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    _fold_dataset = training_script.load_dataset(task, n_fold=n_fold, fold=folds[0],
                                                 max_samples_per_fold=max_samples_per_fold)
    X, y = _fold_dataset.X, _fold_dataset.y
    if 'texts' in _fold_dataset.others:
        texts = _fold_dataset.others['texts']
    else:
        texts = None

    # train_hidden_states = None  # サンプルが重なってて難しいので一旦パス
    val_hidden_states = None
    test_hidden_states = None
    if return_hidden_states:
        val_hidden_states = []
        test_hidden_states = []
        model_dir = Path(model_dir)

    train_indexes = set()  # サンプルが重なっているので順番に意味がない．
    val_indexes = []
    test_indexes = []
    for fold in folds:
        _fold_dataset = training_script.load_dataset(task, n_fold=n_fold, fold=fold,
                                                     max_samples_per_fold=max_samples_per_fold)

        train_indexes = train_indexes.union(set(_fold_dataset.train_indexes))

        _fold_valid_indexes = _fold_dataset.valid_indexes
        if len(set(val_indexes).intersection(set(_fold_valid_indexes))):
            # validは重なりが無いはず．
            raise Exception('Something wrong')
        val_indexes.extend(_fold_valid_indexes)
        if any([val_index.find('train.train') >= 0 for val_index in val_indexes]):
            # HONOKA: ここを通る=ダメ
            raise

        _fold_test_indexes = _fold_dataset.test_indexes
        if len(test_indexes) == 0:
            test_indexes = _fold_test_indexes
        elif test_indexes != _fold_test_indexes:
            # testは完全に一致するはず
            raise Exception('Something wrong')

        if return_hidden_states:
            fold_output_dir = model_dir / f'fold-{fold}'

            val_hidden_states_path = list(
                fold_output_dir.glob('**/val_encoder_outputs/*/hidden_states_pooled.npy')
            )[0]
            val_hidden_states_over_layers = np.load(open(val_hidden_states_path, 'rb'))
            for i_layer, hidden_states_over_samples in enumerate(val_hidden_states_over_layers):
                if len(val_hidden_states) <= i_layer:
                    val_hidden_states.append(hidden_states_over_samples)
                else:
                    val_hidden_states[i_layer] = np.concatenate([
                        val_hidden_states[i_layer],
                        hidden_states_over_samples
                    ])

            test_hidden_states_path = list(
                fold_output_dir.glob('**/test_encoder_outputs/*/hidden_states_pooled.npy')
            )[0]
            test_hidden_states_over_layers = np.load(open(test_hidden_states_path, 'rb'))
            for i_layer, hidden_states_over_samples in enumerate(test_hidden_states_over_layers):
                if cv_average_embedding:
                    if len(test_hidden_states) <= i_layer:
                        test_hidden_states.append(hidden_states_over_samples / len(folds))
                    else:
                        for j_sample, hidden_states in enumerate(hidden_states_over_samples):
                            test_hidden_states[i_layer][j_sample] += hidden_states / len(folds)
                else:
                    if len(test_hidden_states) <= i_layer:
                        test_hidden_states.append(hidden_states_over_samples)
                    else:
                        pass

    if return_hidden_states:
        # XXX: jiantの内部実装に依存してしまっている．
        val_hidden_states_indexed = []
        for i_layer in range(0, len(val_hidden_states)):
            val_hidden_states_indexed.append(
                {idx: vec for idx, vec in zip(val_indexes, val_hidden_states[i_layer])}
            )

        test_hidden_states_indexed = []
        for i_layer in range(0, len(test_hidden_states)):
            test_hidden_states_indexed.append(
                {idx: vec for idx, vec in zip(test_indexes, test_hidden_states[i_layer])}
            )

        val_test_hidden_states_indexed = []
        for i_layer in range(0, len(val_hidden_states_indexed)):
            hidden_states_indexed = val_hidden_states_indexed[i_layer]
            hidden_states_indexed.update(test_hidden_states_indexed[i_layer])
            val_test_hidden_states_indexed.append(hidden_states_indexed)

    # swap
    if valid_to_test_n_fold is not None:
        kf = KFold(n_splits=valid_to_test_n_fold, random_state=0, shuffle=True)
        for i_valid_to_test_fold, (valid_to_valid_indexes, valid_to_test_indexes) in enumerate(kf.split(val_indexes)):
            if i_valid_to_test_fold == valid_to_test_fold:
                valid_split_val_indexes = [val_indexes[i_] for i_ in valid_to_valid_indexes]
                valid_split_test_indexes = [val_indexes[i_] for i_ in valid_to_test_indexes]
        val_indexes = valid_split_val_indexes
        test_indexes = valid_split_test_indexes

    if return_hidden_states:
        _DATASET_CACHE[cache_key] = X, y, train_indexes, val_indexes, test_indexes, val_test_hidden_states_indexed, texts
    else:
        _DATASET_CACHE[cache_key] = X, y, train_indexes, val_indexes, test_indexes, texts

    return _DATASET_CACHE[cache_key]


_SINGLE_MODEL_CACHE: Dict[str, Tuple[TrainScriptConfig, BaseEstimator, BaseEstimator]] = {}


def load_models(task: str,
                base_model_generation_method: str,
                input_dir: Union[str, Path],
                training_script: TrainScriptBase,
                n_fold: int,
                folds: List[int],
                i_models: List[int],
                model_types: List[str],
                valid_to_test_n_fold: Optional[int] = None,
                valid_to_test_fold: Optional[int] = None,
                transform=None,
                max_samples_per_fold: Optional[int] = None) -> Dict[str, BaseEstimator]:
    input_dir = Path(input_dir)

    training_config_paths = [
        Path(path) for path in sorted(glob.glob(str(input_dir / '*/**/config.training_script.json'),
                                                recursive=True))
    ]

    configs = {}
    for training_config_path in training_config_paths:
        logger.info('loading model "%s"', training_config_path)

        generation_params = json.load(open(training_config_path.parent / 'params.model_variants.json'))
        model_type = generation_params['model_type']
        i_model = generation_params['i_model']
        configs[(model_type, i_model)] = training_config_path

    models = {}
    if base_model_generation_method.startswith('adaboost'):
        X, y, train_indexes, valid_indexes, _, _ = load_dataset(
            task,
            training_script,
            n_fold,
            folds,
            valid_to_test_n_fold=valid_to_test_n_fold,
            valid_to_test_fold=valid_to_test_fold,
            max_samples_per_fold=max_samples_per_fold,
        )

        for target_model_type in model_types:
            boosting = build_boosting_inference('adaboost_m1',
                                                X,
                                                y,
                                                train_indexes,
                                                valid_indexes,
                                                num_model_variants=1,
                                                complemental_distribution=False)
            for target_i_model in i_models:
                config_key = (target_model_type, target_i_model)
                if config_key not in configs:
                    raise ConfigNotFoundError(*config_key)
                training_config_path = configs[config_key]
                logger.info('loading model "%s"', training_config_path)

                model_dir = str(training_config_path.parent)
                if model_dir in _SINGLE_MODEL_CACHE:
                    config, model, running_prediction_model = _SINGLE_MODEL_CACHE[model_dir]
                else:
                    config, model, running_prediction_model = training_script.load_model(
                        str(training_config_path.parent),
                        get_running_prediction_model=True)
                _SINGLE_MODEL_CACHE[str(training_config_path.parent)] = config, model, running_prediction_model

                if config.n_fold != n_fold:
                    raise Exception('The conditions models are trained on are different')
                if config.folds != folds:
                    raise Exception('The conditions models are trained on are different')

                boosting.add_model(model, running_prediction_model)
            models[f'adaboost.model_type={target_model_type}'] = boosting
    else:
        for target_model_type in model_types:
            for target_i_model in i_models:
                config_key = (target_model_type, target_i_model)
                if config_key not in configs:
                    raise ConfigNotFoundError(*config_key)
                training_config_path = configs[config_key]
                logger.info('loading model "%s"', training_config_path)

                model_dir = str(training_config_path.parent)
                if model_dir in _SINGLE_MODEL_CACHE:
                    config, model = _SINGLE_MODEL_CACHE[model_dir]
                else:
                    config, model = training_script.load_model(model_dir,
                                                               get_running_prediction_model=False)
                _SINGLE_MODEL_CACHE[model_dir] = config, model

                if config.n_fold != n_fold:
                    raise Exception('The conditions models are trained on are different')
                if config.folds != folds:
                    raise Exception('The conditions models are trained on are different')

                models[f'base_model.model_type={target_model_type}.i_model={target_i_model}'] = model
    if transform is not None:
        return {name: Pipeline([('transform', transform), (name, model)])
                for name, model in models.items()}
    else:
        return models


def build_ensemble_models(
    base_models: Union[Dict[str, BaseEstimator], List[Tuple[str, BaseEstimator]]],
    ensemble_method_name: Optional[str] = None,
    final_classifier_name: Optional[str] = None,
    regulator: Optional[str] = None,
    stack_method: Optional[str] = None,
    n_jobs: Optional[int] = None,
    grid_search: bool = False,
    moe_config: Optional[MoETransformerClassifierConfig] = None,
) -> Dict:

    def get_one(models: Union[Dict[str, BaseEstimator], List[Tuple[str, BaseEstimator]]]):
        if isinstance(models, dict):
            return list(models.values())[0]
        else:
            return models[0][1]

    if isinstance(base_models, list):
        base_models = {ensemble_method_name: model for ensemble_method_name, model in base_models}

    ensemble_models: Dict[str, BaseEstimator] = {}
    if ensemble_method_name is None:
        ensemble_method_names = ['StackingClassifier', 'VotingClassifier']
    else:
        ensemble_method_names = [ensemble_method_name]
    final_classifier_name = final_classifier_name or 'LogisticRegression'

    for ensemble_method_name in ensemble_method_names:
        config = AutoFactory.estimator_config(
            ensemble_method_name,
            stack_method=stack_method or 'auto',
        )

        if hasattr(config, 'n_jobs'):
            config.n_jobs = n_jobs

        if grid_search:
            final_classifier_config = AutoFactory.grid_search_config(
                final_classifier_name,
                n_jobs=n_jobs,
            )
        else:
            final_classifier_config = AutoFactory.estimator_config(final_classifier_name)

        if final_classifier_name == 'LogisticRegression' and regulator is not None:
            if isinstance(final_classifier_config, GridSearchConfig):
                final_classifier_config.search_space.penalty = [regulator]
            else:
                final_classifier_config.penalty = regulator

        if final_classifier_name == 'MoETransformerClassifier':
            if isinstance(final_classifier_config, GridSearchConfig):
                # final_classifier_config.search_space.n_base_models = [len(base_models)]
                # final_classifier_config.search_space.model_name_or_path = [moe_model_name_or_path]
                # final_classifier_config.search_space.tokenizer_name_or_path = [moe_tokenizer_name_or_path]
                raise NotImplementedError()
            else:
                moe_config = moe_config or MoETransformerClassifierConfig()
                for key, val in moe_config.dict().items():
                    setattr(final_classifier_config, key, val)
                final_classifier_config.n_base_models = len(base_models)

        final_classifier = AutoFactory.estimator(final_classifier_config)

        if len(base_models) == 1:
            model = get_one(base_models)
        else:
            model = AutoFactory.estimator(config,
                                          estimators=base_models,
                                          final_estimator=final_classifier)
        ensemble_models[ensemble_method_name] = model
    return ensemble_models


_MTI_BASE_MODEL_DEPENDENT_QUANTITY_CACHE = defaultdict(dict)


def calc_mti(ensemble_model: BaseEstimator,
             base_models: List[BaseEstimator],
             indexes: np.ndarray,
             y,
             # estimation_type: str,
             interaction_order: Optional[int] = None,
             original_base_models: Optional[List[BaseEstimator]] = None,
             concavity_p0: Optional[float] = None) -> Tuple[MTIBounds, MTIBounds]:

    global _MTI_BASE_MODEL_DEPENDENT_QUANTITY_CACHE

    cache_key = tuple([
        id(base_models) if original_base_models is None else id(original_base_models),
        id(indexes),
        id(y),
        interaction_order,
    ])
    cache = _MTI_BASE_MODEL_DEPENDENT_QUANTITY_CACHE[cache_key]
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
    ensemble_preds = ensemble_model.predict(indexes)
    preds_over_models = []
    for model in base_models:
        preds = model.predict(indexes)
        preds_over_models.append(preds)

    labels = [y[idx] for idx in indexes]
    Es = [1 if label != ensemble_preds[i] else 0
          for i, label in enumerate(labels)]

    samples = []
    feature_names = [f'pred_{i_model}' for i_model in range(0, len(base_models))]
    for i_sample, (ensemble_pred, E, label) in enumerate(zip(ensemble_preds, Es, labels)):
        features = []
        for i_model, feature_name in enumerate(feature_names):
            pred = preds_over_models[i_model][i_sample]
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

    omega_cache_key = tuple([
        id(base_models) if original_base_models is None else id(original_base_models),
        id(indexes),
        id(y),
    ])
    omega_cache = _MTI_BASE_MODEL_DEPENDENT_QUANTITY_CACHE[omega_cache_key]
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


def calc_corr_coef_averaged_over_model_pairs(models: List[BaseEstimator],
                                             indexes: np.ndarray,
                                             y) -> Optional[np.ndarray]:
    if len(models) <= 1:
        return None
        # raise Exception('correlation coefficient over single model is not defined.')
    scores_over_models = []
    for model in models:
        method_name, postprocess = get_sample_scoring_function(model)
        scores = postprocess(getattr(model, method_name)(indexes))
        scores_over_models.append((method_name, scores))

    probs_over_models = [
        safe_log(scores) if method_name == 'decision_function' else scores
        for method_name, scores in scores_over_models
    ]

    labels = [y[idx] for idx in indexes]
    corr_coefs = []
    for i_model in range(0, len(probs_over_models)):
        for j_model in range(i_model + 1, len(probs_over_models)):
            corr_coef_abs = np.abs(
                statistics.calc_corr_coef_of_gold_proba(
                    probs_over_models[i_model],
                    probs_over_models[j_model],
                    labels
                )
            )
            _, _, corr_coef_abs, _ = statistics.calc_corr_coef_averaged_over_features(
                probs_over_models[i_model],
                probs_over_models[j_model]
            )

            corr_coefs.append(corr_coef_abs)
    return np.mean(corr_coefs)


def get_final_classifier(model: BaseEstimator) -> Optional[BaseEstimator]:
    real_model = _get_real_model(model)
    if hasattr(real_model, 'final_estimator_'):
        return _get_real_model(real_model.final_estimator_)
    elif hasattr(real_model, 'final_estimator'):
        return _get_real_model(real_model.final_estimator)
    else:
        return None


def get_scores(model: BaseEstimator,
               val_indexes: Sequence,
               test_indexes: Sequence,
               y: np.ndarray,
               texts: Optional[Dict[Any, str]] = None,
               train_indexes: Sequence = None,
               train_sample_weights = None,
               train_prediction_model = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    final_estimator = get_final_classifier(model)
    if isinstance(final_estimator, MoETransformerClassifier):
        val_texts = [texts[idx] for idx in val_indexes]
        final_estimator.texts_for_predict = val_texts

    y_valid = [y[idx] for idx in val_indexes]
    valid_score = model.score(val_indexes, y_valid)

    if isinstance(final_estimator, MoETransformerClassifier):
        test_texts = [texts[idx] for idx in test_indexes]
        final_estimator.texts_for_predict = test_texts

    y_test = [y[idx] for idx in test_indexes]
    test_score = model.score(test_indexes, y_test)

    train_score, train_weighted_score = None, None
    if train_prediction_model is not None:
        y_train = [y[idx] for idx in train_indexes]
        train_score = train_prediction_model.score(train_indexes, y_train)
        train_weighted_score = train_prediction_model.score(train_indexes, y_train, sample_weight=train_sample_weights)
    return valid_score, test_score, train_score, train_weighted_score


def get_model_repr(model: BaseEstimator,
                   indent='') -> str:
    if isinstance(model, Pipeline):
        return '{0}Pipeline(\n{0}    ...,\n    {1})'.format(indent,
                                                            get_model_repr(model.steps[-1][1], indent = indent + '    '))
    elif isinstance(model, (BaseEnsemble, _BaseHeterogeneousEnsemble)) and not isinstance(model, RandomForestClassifier):
        estimators_repr = '\n'.join([
            '{0}    {1}'.format(indent, name)
            for name, estimator in model.estimators
        ])
        if isinstance(model, (StackingClassifier, StackingRegressor)):
            estimators_repr += '\n{0}    meta_estimator={1}'.format(indent, model.final_estimator)
        return '{0}{1}(\n{2})'.format(indent, model.__class__.__name__, estimators_repr)
    elif isinstance(model, GridSearchCV):
        return '{0}GridSearchCV(\n{0}    {1})'.format(indent,
                                                      get_model_repr(model.estimator, indent = indent + '    '))
    else:
        return model.__class__.__name__


def get_results(model_name: str,
                model: BaseEstimator,
                val_indexes, test_indexes,
                y: np.ndarray,
                # is_ensemble=False,
                texts: Optional[Dict[Any, str]] = None,
                train_indexes=None,
                return_model=False,
                mti_interaction_orders: Optional[List[int]] = None,
                concavity_p0: Optional[float] = None,
                no_mti_analysis=False,
                original_base_models: Optional[List[BaseEstimator]] = None) -> Union[
                    EnsembleResults,
                    List[EnsembleResults],
                    Tuple[EnsembleResults, BaseEstimator],
                    Tuple[List[EnsembleResults], BaseEstimator],
]:
    y_valid = [y[idx] for idx in val_indexes]

    # "train.train"というindexが混ざるバグのデバッグ
    # try:
    #     y_valid = [y[idx] for idx in val_indexes]
    # except:
    #     # HONOKA: ここを通る。val_indexesに"train.train"を含むものがあることが問題。
    #     import pprint
    #     print('---- y.keys() ----')
    #     pprint.pprint(list(y.keys()))
    #     print('---- val_indexes ----')
    #     pprint.pprint(val_indexes)

    logger.info('fitting %s', get_model_repr(model))

    final_estimator = get_final_classifier(model)
    if isinstance(final_estimator, MoETransformerClassifier):
        val_texts = [texts[idx] for idx in val_indexes]
        final_estimator.texts_for_fit = val_texts

    model = model.fit(val_indexes, y_valid)

    valid_score, test_score, train_score, train_weighted_score = get_scores(model,
                                                                            val_indexes,
                                                                            test_indexes,
                                                                            y,
                                                                            texts=texts,
                                                                            train_indexes=train_indexes)

    results = EnsembleResults(valid_score=valid_score,
                              test_score=test_score,
                              train_score=train_score,
                              train_weighted_score=train_weighted_score)

    def get_boosting_base_models(model: BaseEstimator) -> Tuple[List[BaseEstimator], List[BaseEstimator], List[np.ndarray]]:
        pipeline = _get_real_model(model, Pipeline)
        preprocess_models = [_model for _model in pipeline.steps[:-1]]

        boosting_inference = _get_real_model(model, BoostingInferenceBase)

        _base_models = []
        _base_running_prediction_models = []
        sample_weights = []
        for t, (boosting_model, boosting_running_prediction_model) in enumerate(zip(boosting_inference.models, boosting_inference.running_prediction_models)):
            sample_weights.append(boosting_inference.get_sample_weights(t, 0))
            _base_models.append(
                Pipeline([*preprocess_models, ('boosting_inference', boosting_model)])
            )
            _base_running_prediction_models.append(
                Pipeline([*preprocess_models, ('boosting_inference', boosting_running_prediction_model)])
            )

        return _base_models, _base_running_prediction_models, sample_weights

    def calc_std(vals):
        if len(vals) <= 1:
            return None
        else:
            return np.std(vals)

    if any([_get_real_model(model, model_type=ensemble_class) is not None
            for ensemble_class in [VotingClassifier, StackingClassifier, BoostingInferenceBase]]):
        is_ensemble = True
    else:
        is_ensemble = False

    real_model = _get_real_model(model)

    is_ensemble = any([_get_real_model(model, model_type=ensemble_class) is not None
                       for ensemble_class in [VotingClassifier, StackingClassifier]])
    is_adaboost = _get_real_model(model, model_type=BoostingInferenceBase) is not None

    best_params = None
    sample_weights_list = None
    train_prediction_models = None
    if is_ensemble:
        if isinstance(real_model, StackingClassifier)\
                and isinstance(real_model.final_estimator_, GridSearchCV):
            best_params = {'final_estimator': model.final_estimator_.best_params_}
        base_models = real_model.estimators_
    elif is_adaboost:
        base_models, train_prediction_models, sample_weights_list = get_boosting_base_models(model)
    else:
        grid_search_model = _get_real_model(model, model_type=GridSearchCV)
        if grid_search_model is not None:
            best_params = grid_search_model.best_params_

        base_models = [real_model]

    # base_models_valid_corr_coeff = calc_corr_coef_averaged_over_model_pairs(base_models, val_indexes, y)
    # base_models_test_corr_coeff = calc_corr_coef_averaged_over_model_pairs(base_models, test_indexes, y)
    base_models_valid_corr_coeff = None
    base_models_test_corr_coeff = None

    train_prediction_models = train_prediction_models or [None] * len(base_models)
    sample_weights_list = sample_weights_list or [None] * len(base_models)
    # XXX: baselineモデルで，weightedなerrorを計測する．
    if len(sample_weights_list) >= 2:
        sample_weights_list[0] = sample_weights_list[1]
    base_model_scores = [
        get_scores(
            base_model,
            val_indexes,
            test_indexes,
            y,
            train_indexes=train_indexes,
            train_sample_weights=sample_weights,
            train_prediction_model=train_prediction_model,
        )
        for base_model, train_prediction_model, sample_weights in zip(base_models, train_prediction_models, sample_weights_list)
    ]

    base_model_valid_scores = [valid_score for valid_score, *_ in base_model_scores]
    base_model_valid_score_average = np.mean(base_model_valid_scores)
    base_model_valid_score_std = calc_std(base_model_valid_scores)

    base_model_test_scores = [test_score for _, test_score, *_ in base_model_scores]
    base_model_test_score_average = np.mean(base_model_test_scores)
    base_model_test_score_std = calc_std(base_model_test_scores)

    base_model_train_scores = [train_score for _, _, train_score, *_ in base_model_scores]
    base_model_train_score_average = None
    base_model_train_score_std = None
    if all([score is not None for score in base_model_train_scores]):
        base_model_train_score_average = np.mean(base_model_train_scores)
        base_model_train_score_std = calc_std(base_model_train_scores)

    base_model_train_weighted_scores = [train_weighted_score for *_, train_weighted_score in base_model_scores]
    base_model_train_weighted_score_average = None
    base_model_train_weighted_score_std = None
    if all([score is not None for score in base_model_train_weighted_scores]):
        base_model_train_weighted_score_average = np.mean(base_model_train_weighted_scores)
        base_model_train_weighted_score_std = calc_std(base_model_train_weighted_scores)

    all_results = []
    _mti_interaction_orders = mti_interaction_orders or [None]
    for mti_interaction_order in _mti_interaction_orders:
        logger.info('---------------- mti interaction order %s --------------------', mti_interaction_order)
        _results = results.copy()

        logger.info('------- calculating mti .. --------')
        valid_mti_bounds, valid_mti_bounds_previous_research = None, None
        if no_mti_analysis:
            test_mti_bounds, test_mti_bounds_previous_research = MTIBounds(), MTIBounds()
        else:
            test_mti_bounds, test_mti_bounds_previous_research = calc_mti(
                model,
                base_models,
                test_indexes,
                y,
                interaction_order=mti_interaction_order,
                original_base_models=original_base_models,
                concavity_p0=concavity_p0,
            )

        _results.best_params = best_params
        if best_params is not None:
            logger.info('best params: %s', pformat(best_params))

        _results.base_model_valid_scores = base_model_valid_scores
        _results.base_model_valid_score_average = base_model_valid_score_average
        _results.base_model_valid_score_std = base_model_valid_score_std

        _results.base_model_test_scores = base_model_test_scores
        _results.base_model_test_score_average = base_model_test_score_average
        _results.base_model_test_score_std = base_model_test_score_std

        _results.base_model_train_scores = base_model_train_scores
        _results.base_model_train_score_average = base_model_train_score_average
        _results.base_model_train_score_std = base_model_train_score_std

        _results.base_model_train_weighted_scores = base_model_train_weighted_scores
        _results.base_model_train_weighted_score_average = base_model_train_weighted_score_average
        _results.base_model_train_weighted_score_std = base_model_train_weighted_score_std

        _results.base_model_valid_corr_coef = base_models_valid_corr_coeff
        _results.base_model_test_corr_coef = base_models_test_corr_coeff

        _results.valid_mti_bounds = valid_mti_bounds
        _results.test_mti_bounds = test_mti_bounds

        _results.valid_mti_bounds_previous_research = valid_mti_bounds_previous_research
        _results.test_mti_bounds_previous_research = test_mti_bounds_previous_research

        all_results.append(_results)

    if return_model:
        return all_results, model
    else:
        return all_results


class IndexToXTransform(BaseEstimator):

    def __init__(self, X):
        self.X = X

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, y=None):
        return [self.X[idx] for idx in X]


def load_setup(input_dir: Union[str, Path]) -> Dict:
    input_dir = Path(input_dir)
    base_model_generation_config = json.load(open(input_dir / 'config.base_model_generation.json', 'r'))
    generate_base_models_config = json.load(open(input_dir / 'config.generate_base_models.json', 'r'))
    return {
        'n_fold': base_model_generation_config['n_fold'],
        'folds': base_model_generation_config['folds'],
        'task_name': base_model_generation_config['task_name'],
        'method': generate_base_models_config['method'],
        'model_types': base_model_generation_config['model_types'],
        'num_models_per_type': generate_base_models_config['num_models_per_type'],
        'start_seed': base_model_generation_config['start_seed'],
        # 'train_from_scratch': base_model_generation_config['train_from_scratch'],
        'train_from_scratch': base_model_generation_config.get('train_from_scratch', None),
        'max_samples_per_fold': base_model_generation_config.get('max_samples_per_fold', None),
    }


def get_single_model_results(task: str,
                             base_model_generation_method: str,
                             input_dir: Union[str, Path],
                             training_script: TrainScriptConfig,
                             n_fold: int,
                             folds: List[int],
                             model_type: str,
                             i_model: int,
                             val_indexes: Sequence,
                             test_indexes: Sequence,
                             X: np.ndarray,
                             y: np.ndarray,
                             train_indexes=None,
                             no_mti_analysis=False,
                             concavity_p0: Optional[float] = None,
                             max_samples_per_fold: Optional[int] = None) -> List[EnsembleResults]:
    logger.info('-------------------------------------- single fine-tuned models -------------------------------------------')
    index_to_X_transform = IndexToXTransform(X)

    all_results: List[EnsembleResults] = []
    try:
        models = load_models(task,
                             base_model_generation_method,
                             input_dir,
                             training_script,
                             n_fold,
                             folds,
                             model_types=[model_type],
                             i_models=[i_model],
                             transform=index_to_X_transform,
                             max_samples_per_fold=max_samples_per_fold)
    except (ConfigNotFoundError, UncompletedTrainingError) as e:
        logger.warning('Return empty results since some of the model config not found. Maybe, training is not done.')
        logger.warning('The Exception is the following: %s', str(e))
        return []
    if len(models) != 1:
        raise Exception('Something wrong')

    _, model = list(models.items())[0]
    model_name = f'{model_type}__i_model={i_model}'
    results = get_results(model_name,
                          model,
                          val_indexes,
                          test_indexes,
                          y,
                          mti_interaction_orders=[None],
                          train_indexes=train_indexes,
                          no_mti_analysis=no_mti_analysis,
                          concavity_p0=concavity_p0)[0]
    results.ensemble_method = 'single'
    results.model_type = model_name
    results.num_models_per_type = 1
    all_results.append(results)
    return all_results


def get_ensemble_model_results(task: str,
                               base_model_generation_method: str,
                               input_dir: Union[str, Path],
                               training_script: TrainScriptBase,
                               n_fold: int,
                               folds: List[int],
                               _num_models_per_type: int,
                               model_type_set: List[str],
                               model_type_set_name: str,
                               val_indexes: Sequence,
                               test_indexes: Sequence,


                               X: np.ndarray,
                               y: np.ndarray,

                               texts: Optional[Dict[Any, str]] = None,

                               valid_to_test_n_fold: Optional[int] = None,
                               valid_to_test_fold: Optional[int] = None,
                               max_samples_per_fold: Optional[int] = None,

                               meta_estimator_set: str = 'middle',
                               stack_method: Optional[str] = None,
                               stack_method_is_predict_proba_if_log_reg=False,

                               no_mti_analysis=False,
                               concavity_p0: Optional[float] = None,

                               moe_config: Optional[MoETransformerClassifierConfig] = None,
                               moe_meta_estimator_rename: Optional[str] = None,

                               train_indexes=None,
                               n_jobs: Optional[int] = None,
                               grid_search: bool = False) -> List[EnsembleResults]:
    logger.info('-------------------------------------- ensemble models -------------------------------------------')
    index_to_X_transform = IndexToXTransform(X)

    try:
        base_models = load_models(task,
                                  base_model_generation_method,
                                  input_dir,
                                  training_script,
                                  n_fold,
                                  folds,
                                  i_models=list(range(0, _num_models_per_type)),

                                  valid_to_test_n_fold=valid_to_test_n_fold,
                                  valid_to_test_fold=valid_to_test_fold,
                                  model_types=model_type_set,
                                  transform=index_to_X_transform,
                                  max_samples_per_fold=max_samples_per_fold)
    except (ConfigNotFoundError, UncompletedTrainingError) as e:
        logger.warning('Return empty results since some of the model config not found. Maybe, training is not done.')
        logger.warning('The exception is the following: %s', str(e))
        return []

    if meta_estimator_set == 'only_voting':
        args: List[Tuple[Optional[str], Optional[str], Optional[str]]] = [
            ('VotingClassifier', None, None),
        ]
    elif meta_estimator_set == 'small':
        args = [
            ('VotingClassifier', None, None),
            ('StackingClassifier', 'LogisticRegression', None),
        ]
    elif meta_estimator_set == 'middle':
        args = [
            ('VotingClassifier', None, None),
            ('StackingClassifier', 'LogisticRegression', None),
            ('StackingClassifier', 'SVC', None),
            ('StackingClassifier', 'RandomForestClassifier', None),
        ]
    elif meta_estimator_set == 'large':
        args = [
            ('VotingClassifier', None, None),
            # ('StackingClassifier', 'LogisticRegression', 'l1'),
            # ('StackingClassifier', 'LogisticRegression', 'l2'),
            ('StackingClassifier', 'LogisticRegression', None),
            ('StackingClassifier', 'SVC', None),
            # ('StackingClassifier', 'LinearSVC', None),
            ('StackingClassifier', 'KNeighborsClassifier', None),
            ('StackingClassifier', 'RandomForestClassifier', None),
            ('StackingClassifier', 'DecisionTreeClassifier', None),
            # ('StackingClassifier', 'GaussianProcessClassifier', None),   # メモリ消費量大?
        ]
    elif meta_estimator_set == 'moe.only_moe':
        args = [
            ('StackingClassifier', 'MoETransformerClassifier', None),
        ]
    elif meta_estimator_set == 'moe.small':
        args = [
            ('StackingClassifier', 'MoETransformerClassifier', None),
            ('StackingClassifier', 'LogisticRegression', None),
            ('VotingClassifier', None, None),
        ]
    else:
        raise ValueError()

    all_results: List[EnsembleResults] = []
    for ensemble_method_name, final_classifier_name, regulator in args:
        logger.info('---- start get results of %s %s %s ----',
                    str(ensemble_method_name),
                    str(final_classifier_name),
                    str(regulator))
        if stack_method_is_predict_proba_if_log_reg and final_classifier_name == 'LogisticRegression':
            _stack_method = 'predict_proba'
        else:
            _stack_method = stack_method

        ensemble_models = build_ensemble_models(
            base_models,
            ensemble_method_name=ensemble_method_name,
            final_classifier_name=final_classifier_name,
            regulator=regulator,
            stack_method=_stack_method,
            n_jobs=n_jobs,
            grid_search=grid_search,
            moe_config=moe_config,
        )

        if final_classifier_name == 'MoETransformerClassifier' and moe_meta_estimator_rename is not None:
            final_classifier_rename = moe_meta_estimator_rename
        else:
            final_classifier_rename = final_classifier_name

        for ensemble_name, ensemble_model in ensemble_models.items():
            mti_interaction_orders = [None, 3, 2]

            results_list, model = get_results(ensemble_name,
                                              ensemble_model,
                                              val_indexes, test_indexes, y,
                                              texts=texts,
                                              # is_ensemble=is_really_ensemble,
                                              train_indexes=train_indexes,
                                              return_model=True,
                                              mti_interaction_orders=mti_interaction_orders,
                                              original_base_models=base_models,
                                              no_mti_analysis=no_mti_analysis,
                                              concavity_p0=concavity_p0)
            for results in results_list:
                results.final_classifier_name = final_classifier_rename
                results.regulator = regulator
                results.ensemble_method = ensemble_name
                results.model_type = model_type_set_name
                results.num_models_per_type = _num_models_per_type

                if ensemble_name == 'StackingClassifier'\
                        and final_classifier_name == 'LogisticRegression'\
                        and len(base_models) >= 2:
                    stacking_model = _get_real_model(model, model_type=StackingClassifier)
                    logreg_model = _get_real_model(stacking_model.final_estimator_, model_type=LogisticRegression)
                    weights = logreg_model.coef_.T
                    weights = weights.tolist()
                    results.base_model_weights = weights

                all_results.append(results)
    return all_results
