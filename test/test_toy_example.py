from pprint import pprint
import random
from typing import List

import numpy as np
from ensemble_metrics import compute_metrics

random.seed(0)


# -- make toy datset
klasses = [0, 1, 2]
num_samples = 10000
labels = np.array(random.choices(klasses, k=num_samples))
features = np.array([-1] * num_samples)  # anything is ok because the toy model do not use this feature


# -- make predictions with toy models

class SpecifiedErrorRateModel:

    def __init__(self, error_rate: float):
        self.error_rate = error_rate

    def predict(self, X: np.array) -> np.array:
        preds: List[int] = []
        for label in labels:
            if random.random() < self.error_rate:
                wrong_labels = [k for k in klasses if k != label]
                preds.append(random.choice(wrong_labels))
            else:
                preds.append(label)
        return np.array(preds)


num_base_models = 5
base_model_error_rate = 0.3

base_models = [SpecifiedErrorRateModel(base_model_error_rate) for _ in range(num_base_models)]
base_model_preds = [base_model.predict(features) for base_model in base_models]


def vote(i_sample: int) -> int:
    _base_model_preds = [base_model_preds[i_model][i_sample] for i_model in range(num_base_models)]
    klass_counts = {klass_: _base_model_preds.count(klass_) for klass_ in klasses}
    random.shuffle(klasses)  # for random choice on tie cases
    return max(klass_counts, key=klass_counts.get)


ensemble_preds = np.array([vote(i_sample) for i_sample in range(num_samples)])


# -- calculate metrics
MTI_k = 3   # "k" of MTK_k in the paper
p0 = base_model_error_rate   # "p_0" in the paper

metrics = compute_metrics(
    labels,
    base_model_preds,
    ensemble_preds,
    MTI_k=MTI_k,
    p0=p0,
)


# -- the three metrics and the ensemble strength
print('relevance:', metrics.relevance)
print('redundancy:', metrics.redundancy)
print('combloss:', metrics.combloss)

# -- the error rate lower bounds
print('loose_LB_wo_combloss:', metrics.loose_LB_wo_combloss)
print('tight_LB_wo_combloss:', metrics.tight_LB_wo_combloss)
print('tight_LB_w_combloss:', metrics.tight_LB_w_combloss)
