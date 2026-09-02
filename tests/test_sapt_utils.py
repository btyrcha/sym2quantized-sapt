from sympy import symbols, latex

from sym2quantized_sapt.sapt_utils import get_a_operator, get_b_operator


def test_get_a_operator_for_one_electron():
    """a^{p_0}_{q_0} = ad(p_0) a(q_0)"""
    reference = r"a^\dagger_{p_0} a_{q_0}"

    tested_str = latex(get_a_operator(n=1))

    assert reference == tested_str


def test_get_a_operator_reverses_the_annihilators():
    """the annihilation part runs backwards, so the two electron operator
    is ad(p_0) ad(p_1) a(q_1) a(q_0) and not ... a(q_0) a(q_1)"""
    reference = r"a^\dagger_{p_0} a^\dagger_{p_1} a_{q_1} a_{q_0}"

    tested_str = latex(get_a_operator(n=2))

    assert reference == tested_str


def test_get_b_operator_for_one_electron():
    """b^{r_0}_{s_0} = bd(r_0) b(s_0)"""
    reference = r"b^\dagger_{r_0} b_{s_0}"

    tested_str = latex(get_b_operator(n=1))

    assert reference == tested_str


def test_get_b_operator_reverses_the_annihilators():
    reference = r"b^\dagger_{r_0} b^\dagger_{r_1} b_{s_1} b_{s_0}"

    tested_str = latex(get_b_operator(n=2))

    assert reference == tested_str


def test_get_a_operator_with_explicit_indicies():
    """`n` wins over the explicit indicies, so they are only used when it
    is falsy - this is what the "if given ignors" in the docstring means"""
    reference = r"a^\dagger_{p} a^\dagger_{r} a_{s} a_{q}"

    p, q, r, s = symbols("p q r s", is_molA=True)

    tested_str = latex(get_a_operator([p, r], [q, s], n=0))

    assert reference == tested_str


def test_get_b_operator_with_explicit_indicies():
    reference = r"b^\dagger_{p} b_{q}"

    p, q = symbols("p q", is_molB=True)

    tested_str = latex(get_b_operator([p], [q], n=0))

    assert reference == tested_str


def test_get_a_operator_ignores_indicies_when_n_is_given():
    """the explicit indicies are silently discarded, not merged"""
    p, q = symbols("p q", is_molA=True)

    with_indicies = latex(get_a_operator([p], [q], n=1))
    without_indicies = latex(get_a_operator(n=1))

    assert with_indicies == without_indicies
