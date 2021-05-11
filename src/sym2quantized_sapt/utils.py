from functools import wraps
from time import process_time

from sympy import latex
from sympy.core import Expr

# NOTE: https://dev.to/po5i/python-decorator-to-measure-function-s-execution-time-4d26
def timeit(func):
    """timing decorator"""

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(process_time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(process_time() * 1000)) - start
            print(
                f"Total execution time {func.__name__}: {end_ if end_ > 0 else 0} ms"
            )

    return _time_it


def format_expr(expression: Expr) -> str:
    """formats expression to str encoded LaTeX

    Args:
        expression (Expr): expression to be encoded

    Returns:
        str: expression LaTeX string
    """
    expression_repr = ""
    for arg in expression.args:
        latex_str = latex(arg)
        # substract sign
        if latex_str[0] == "-":
            expression_repr += f"& {latex_str} \\\\ \n"
        # add sign
        else:
            expression_repr += f"& + {latex_str} \\\\ \n"

    return expression_repr
