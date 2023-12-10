"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable, Tuple, Sequence

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # Implement for Task 0.1.
    return x * y


def mul_back(x: float, y: float, d: float) -> Tuple[float, float]:
    return d * y, d * x


def id(x: float) -> float:
    "$f(x) = x$"
    # Implement for Task 0.1.
    return x


id_back = id


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # Implement for Task 0.1.
    return x + y


def add_back(d: float) -> Tuple[float, float]:
    return d, d


def neg(x: float) -> float:
    "$f(x) = -x$"
    # Implement for Task 0.1.
    return -1.0 * x


def neg_back(d: float) -> float:
    return -1.0 * d


def no_grad(num_inputs) -> float | Sequence[float]:
    if num_inputs == 1:
        return 0.0
    else:
        return tuple(0.0 for _ in range(num_inputs))


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # Implement for Task 0.1.
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # Implement for Task 0.1.
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # Implement for Task 0.1.
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # Implement for Task 0.1.
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # Implement for Task 0.1.
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0.0 else math.exp(x) / (1.0 + math.exp(x))


def sigmoid_back(x: float, d: float) -> float:
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x) * d


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # Implement for Task 0.1.
    return x if x > 0.0 else 0.0


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # Implement for Task 0.1.
    return d if x > 0.0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # Implement for Task 0.1.
    return d / (x + EPS) if x == 0.0 else d / x


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def exp_back(x: float, d: float) -> float:
    return exp(x) * d


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # Implement for Task 0.1.
    return 1.0 / (x + EPS) if x == 0.0 else 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # Implement for Task 0.1.
    inv_x = 1.0 / (x + EPS) if x == 0.0 else 1.0 / x
    return -d * inv_x * inv_x



# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # Implement for Task 0.3.
    def _map_fn(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return _map_fn


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # Implement for Task 0.3.
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # Implement for Task 0.3.
    def _zipWith_fn(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith_fn


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # Implement for Task 0.3.
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # Implement for Task 0.3.
    def _reduce_fn(ls: Iterable[float]) -> float:
        acc = start
        for x in ls:
            acc = fn(x, acc)
        return acc

    return _reduce_fn


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # Implement for Task 0.3.
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # Implement for Task 0.3.
    return reduce(mul, 1)(ls)
