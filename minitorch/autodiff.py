from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import deque, defaultdict
# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Implement for Task 1.1.
    h = epsilon / 2
    return (f(*vals[:arg], vals[arg] + h, *vals[arg+1:]) - f(*vals[:arg], vals[arg] - h, *vals[arg+1:])) / epsilon

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Implement for Task 1.4.
    def recursive_topological_sort(v=variable, ordering=deque(), visited=set()):
        visited.add(v.unique_id)
        for p in v.parents:
            if p.unique_id not in visited:
                recursive_topological_sort(p, ordering, visited)
        ordering.appendleft(v)
        return ordering
    
    def iterative_topological_sort(start=variable):    
        status = defaultdict(int)
        NEW = 0
        ENQUEUED = 1
        VISITED = 2
        stack = deque([start])
        status[start.unique_id] = ENQUEUED
        ordering = deque()
        while stack:
            v = stack[-1]
            if status[v.unique_id] == ENQUEUED:
                status[v.unique_id] = VISITED
                for p in v.parents:
                    if status[p.unique_id] == NEW:
                        stack.append(p)
                        status[p.unique_id] = ENQUEUED
            elif status[v.unique_id] == VISITED:
                stack.pop()
                ordering.appendleft(v)
        return ordering
    
    def kahn_toplogical_sort(start=variable):
        visited = set()
        indegree = defaultdict(int)
        indegree[start.unique_id] = 0
        queue = deque([start])
        while queue:
            v = queue.pop()
            if v.unique_id not in visited:
                visited.add(v.unique_id)
                for p in v.parents:
                    indegree[p.unique_id] += 1
                    queue.append(p)
        ordering = []
        queue = deque([start])
        while queue:
            v = queue.popleft()
            ordering.append(v)
            for p in v.parents:
                indegree[p.unique_id] -= 1
                if indegree[p.unique_id] == 0:
                    queue.append(p)
        return ordering
    
    return iterative_topological_sort()       

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Implement for Task 1.4.
    derivs = defaultdict(int)
    derivs[variable.unique_id] = deriv
    for v in topological_sort(variable):
        d_output = derivs[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(d_output)
        elif not v.is_constant():
            for input_var, local_deriv in v.chain_rule(d_output):
                derivs[input_var.unique_id] += local_deriv
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
