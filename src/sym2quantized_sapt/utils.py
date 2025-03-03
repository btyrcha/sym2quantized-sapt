from functools import wraps
from time import process_time

from sympy import latex, S
from sympy.core.add import Add


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


def format_expr(expression: Add) -> str:
    """formats expression to str encoded LaTeX

    Args:
        expression (Expr): expression to be encoded

    Returns:
        str: expression LaTeX string
    """

    if expression is S.Zero:
        return "0"

    expression_repr = ""
    for arg in expression.args:
        latex_str = latex(arg)
        # substract sign
        if latex_str[0] == "-":
            expression_repr += f"& {latex_str} \\\\ \n"
        # add sign
        else:
            expression_repr += f"& + {latex_str} \\\\ \n"

    expression_repr = expression_repr.replace("v_A", "(v_A)")
    expression_repr = expression_repr.replace("v_B", "(v_B)")
    expression_repr = expression_repr.replace("o_A", "(o_A)")
    expression_repr = expression_repr.replace("o_B", "(o_B)")
    expression_repr = expression_repr.replace("t_A", "(t_A)")
    expression_repr = expression_repr.replace("t_B", "(t_B)")
    expression_repr = expression_repr.replace("s^", "S^")
    expression_repr = expression_repr.replace("1.0", "1")
    expression_repr = expression_repr.replace("2.0", "2")
    expression_repr = expression_repr.replace("4.0", "4")
    expression_repr = expression_repr.replace("8.0", "8")

    return expression_repr
