from typing import List, Tuple, Callable
from lin_alg import LinearAlgebra
import math

Vector = List[float]
Matrix = List[List[float]]



class Statistics:
    
    def mean(xs:List[float]) -> float:
    return sum(xs) / len(xs)

    def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs)//2]

    def _median_even(xs: List[float]) -> float:
    sorted_values = sorted(xs)
    hi_mid_point = len(xs)//2
    return (sorted_values[hi_mid_point] + sorted_values[hi_mid_point-1])/2

    def median(v: List[float]) -> float:
    return _median_even(v) if len(v)%2 == 0 else _median_odd(v) 

    def quantile(xs: List[float], p: float) -> float:
    p_index = int(p*len(xs))
    return sorted(xs)[p_index]

    def mode(x:List[float]) -> List[float]:
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count  == max_count]

    def data_range(xs:List[float]) -> float:
    return max(xs) - min(xs)

    def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

    def variance(xs: List[float]) -> float:
    assert len(xs) >= 2, "variance requires at least two elements"
    n = len(xs)
    deviations = de_mean(xs)
    return LinearAlgebra.sum_of_squares(deviations) / (n - 1)

    def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))

    def innterquartile_range(xs:List[float]) -> float:
    return quantile(xs, .75) - quantile(xs, .25)

    def covariance(xs: List[float], ys: List[float]) ->float:
    assert len(xs) == len(ys), "Vectors must have same number of elements!"
    return LinearAlgebra.dot(de_mean(xs), de_mean(ys))/ (len(xs) -1)

    def correlation(xs:List[float], ys:List[float]) -> float:
    std_x = standard_deviation(xs)
    std_y = standard_deviation(ys)
    if std_x > 0 and std_y > 0:
        return covariance(xs, ys)/std_x/std_y
    else:
        return 0