from pprint import pprint
import logging

from ensemble_metrics.frequency_distribution import FrequencyDistribution
from ensemble_metrics.metrics import (
    calc_multi_information,
    calc_conditional_multi_information,
    safe_log2,
)
from numpy.testing import assert_almost_equal

logger = logging.getLogger(__name__)


def test():
    def get_stats(samples):
        MTI_k = 3
        samples = [dict(sample) for sample in samples]
        print('\n----------- samples -------------')
        pprint(samples)

        dist = FrequencyDistribution(samples, count_threshold=0)
        print('\n----------- P(x1, x2, y) -------------')
        print(dist)

        print('\n----------- P(x1, x2 | y=0) -------------')
        conditional = dist.conditional({'y': 0})
        print(conditional)

        print('\n----------- P(x1) -------------')
        marginal = dist.marginal(include_variables=['x1'])
        print(marginal)

        print('\n----------- P(y) -------------')
        marginal = dist.marginal(include_variables=['y'])
        print(marginal)

        print('\n------------ MI(x1, x2, y) --------------')
        multi_information = calc_multi_information(dist)
        print(multi_information)

        print('\n------------ MI(x1, x2, y) MTI_k={0} --------------'.format(MTI_k))
        multi_information_k3 = calc_multi_information(dist, order=MTI_k)
        print(multi_information_k3)
        assert_almost_equal(multi_information, multi_information_k3)

        print('\n------------ MI(x1, x2 | y) --------------')
        conditional_multi_information = calc_conditional_multi_information(dist, ['y'])
        print(conditional_multi_information)

        print('\n------------ MI(x1, x2 | y) MTI_k={0} --------------'.format(MTI_k))
        conditional_multi_information_k3 = calc_conditional_multi_information(dist, ['y'], order=MTI_k)
        print(conditional_multi_information)

        assert_almost_equal(conditional_multi_information_k3, conditional_multi_information)

        return multi_information, multi_information_k3, conditional_multi_information, conditional_multi_information_k3

    print('\n\n======================== variables without any correlation =======================')
    get_stats([
        (('x1', 0), ('x2', 0), ('y', 0)),

        (('x1', 1), ('x2', 0), ('y', 0)),
        (('x1', 0), ('x2', 1), ('y', 0)),
        (('x1', 0), ('x2', 0), ('y', 1)),

        (('x1', 1), ('x2', 1), ('y', 0)),
        (('x1', 1), ('x2', 0), ('y', 1)),
        (('x1', 0), ('x2', 1), ('y', 1)),

        (('x1', 1), ('x2', 1), ('y', 1)),
    ])

    print('\n\n======================== weakly correlated variables =======================')
    get_stats([
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 1), ('x2', 0), ('y', 0)),
        (('x1', 1), ('x2', 1), ('y', 0)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 0), ('x2', 1), ('y', 1)),
        (('x1', 0), ('x2', 0), ('y', 1)),
    ])

    print('\n\n======================== strongly correlated variables =======================')
    mti, mti_k3, cond_mti, cond_mti_k3 = get_stats([
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 1), ('x2', 1), ('y', 0)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 0), ('x2', 0), ('y', 1)),
    ])
    # print(0.75 * safe_log2(0.75 / (0.75 * 0.75)))
    cond_mti_reference = 0.75 * safe_log2(0.75 / (0.75 * 0.75))\
        + 0.25 * safe_log2(0.25 / (0.25 * 0.25))
    assert_almost_equal(cond_mti, cond_mti_reference)

    print('\n\n======================== perfectly correlated variables =======================')
    mti, mti_k3, cond_mti, cond_mti_k3 = get_stats([
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 0), ('x2', 0), ('y', 0)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 1), ('x2', 1), ('y', 1)),
        (('x1', 1), ('x2', 1), ('y', 1)),
    ])
    mti_reference = safe_log2(4)
    assert_almost_equal(mti, mti_reference)
    print('MI(x1, x2 | y) = 0.0 is OK since the joint distribution of (x1, x2) is zero from the beginning.')


if __name__ == '__main__':
    test()
