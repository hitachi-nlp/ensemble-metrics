# Ensemble Learning Theory
This is the official codebase for the paper ICML(2022) paper [Rethinking Fano’s Inequality in Ensemble Learning](https://arxiv.org/abs/2205.12683).
The paper proposes a theory about Ensemble Learning that evaluates a given ensemble system by a well-grounded set of metrics: relevance(~accuracy), redundancy(~diversity) and combination loss.
This repo contains the souce codes for calculating the ensemble metrics proposed in the paper.

## Installation
```console
git clone https://github.com/hitachi-nlp/ensemble-metrics.git
cd ensemble-metrics
pip install -e .
```

## How to use
See [test_toy.py](./test/test_toy.py).

## Citation
Please cite our paper as:
```
@InProceedings{pmlr-v162-morishita22a,
  title = 	 {Rethinking Fano’s Inequality in Ensemble Learning},
  author =       {Morishita, Terufumi and Morio, Gaku and Horiguchi, Shota and Ozaki, Hiroaki and Nukaga, Nobuo},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {15976--16016},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/morishita22a/morishita22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/morishita22a.html},
  abstract = 	 {We propose a fundamental theory on ensemble learning that evaluates a given ensemble system by a well-grounded set of metrics. Previous studies used a variant of Fano’s inequality of information theory and derived a lower bound of the classification error rate on the basis of the accuracy and diversity of models. We revisit the original Fano’s inequality and argue that the studies did not take into account the information lost when multiple model predictions are combined into a final prediction. To address this issue, we generalize the previous theory to incorporate the information loss. Further, we empirically validate and demonstrate the proposed theory through extensive experiments on actual systems. The theory reveals the strengths and weaknesses of systems on each metric, which will push the theoretical understanding of ensemble learning and give us insights into designing systems.}
}
```

## License
MIT
