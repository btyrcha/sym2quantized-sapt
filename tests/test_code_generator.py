import pytest
import string

from sympy import symbols, Dummy, Rational
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.double_fermi_vac import (
    wicks_double_vac,
    substitute_dummies_double_vac,
    CreateFermion_A,
    CreateFermion_B,
    AnnihilateFermion_A,
    AnnihilateFermion_B,
)
from sym2quantized_sapt.sapt_utils import get_V_operator
from sym2quantized_sapt.code_generator import generate_einsum


def test_single_self_contraction():
    reference = """+np.einsum("kk", A_kk)"""

    k = symbols("k", below_fermi=True)
    A = DoubleVacuumTensorSymbol("A", (k,), (k,))

    tested_expr = generate_einsum(A)

    assert reference == tested_expr


def test_doucle_self_contraction():
    reference = """+np.einsum("klkl", A_klkl)"""

    k, l = symbols("k l", below_fermi=True)
    A = DoubleVacuumTensorSymbol(
        "A",
        (
            k,
            l,
        ),
        (
            k,
            l,
        ),
    )

    tested_expr = generate_einsum(A)

    assert reference == tested_expr


def test_partial_self_contraction():
    reference = """+np.einsum("kmkl->ml", A_kmkl)"""

    k, l, m = symbols("k l m", below_fermi=True)
    A = DoubleVacuumTensorSymbol(
        "A",
        (
            k,
            l,
        ),
        (
            k,
            m,
        ),
    )

    tested_expr = generate_einsum(A)

    assert reference == tested_expr


def test_two_eris():
    reference = """-np.einsum("abrs,rsab", v_abrs, v_rsab)"""

    a = symbols("a", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    v_ijab = DoubleVacuumTensorSymbol(
        "v",
        (
            a,
            b,
        ),
        (
            i,
            j,
        ),
    )
    v_abij = DoubleVacuumTensorSymbol(
        "v",
        (
            i,
            j,
        ),
        (
            a,
            b,
        ),
    )

    tested_expr = generate_einsum((-1.0) * v_ijab * v_abij)

    assert reference == tested_expr


def test_matrix_multiplication():
    reference = """+np.einsum("kl,lm->km", A_kl, B_lm)"""

    k, l, m = symbols("k l m")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))

    tested_str = generate_einsum(A * B)

    assert reference == tested_str


def test_index_name_collision():
    # `p_1` must not be renamed to `c`, which is already taken by another index
    reference = """+np.einsum("cd,dc", A_cp, B_pc)"""

    c, p_1 = symbols("c p_1")

    A = DoubleVacuumTensorSymbol("A", (p_1,), (c,))
    B = DoubleVacuumTensorSymbol("B", (c,), (p_1,))

    tested_str = generate_einsum(A * B)

    assert reference == tested_str


@pytest.mark.xfail(
    strict=True,
    reason="`pretty_indices` already names the dummies in the psi4numpy "
    "alphabet, and `generate_einsum` renames them a second time: the "
    "particle `r` stays `r` while the hole `a` also becomes `r`, so the "
    "four indices collapse to two and the einsum contracts the wrong pairs",
)
def test_psi4_indices_name_collision():
    # NOTE: pretty_indices in dummies substitution is not compatible with
    # code generation - see the docstring of `substitute_dummies_double_vac`
    reference = """+np.einsum("rsab,abrs", t_rsab, v_abrs)"""

    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    b = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
    j = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

    T11 = (
        DoubleVacuumTensorSymbol("t", (i, j), (a, b))
        * CreateFermion_A(a)
        * AnnihilateFermion_A(i)
        * CreateFermion_B(b)
        * AnnihilateFermion_B(j)
    )
    V = get_V_operator()

    expr = V * T11
    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
    expr = substitute_dummies_double_vac(
        expr,
        pretty_indices={
            "above_molA": "r",
            "above_molB": "s",
            "below_molA": "a",
            "below_molB": "b",
        },
    )

    tested_expr = generate_einsum(expr)

    assert reference == tested_expr


def test_multidigit_index_names():
    # `p_1` must not be substituted inside `p_10`
    reference = """+np.einsum("cd,dc", A_pp, B_pp)"""

    p_1, p_10 = symbols("p_1 p_10")

    A = DoubleVacuumTensorSymbol("A", (p_1,), (p_10,))
    B = DoubleVacuumTensorSymbol("B", (p_10,), (p_1,))

    tested_str = generate_einsum(A * B)

    assert reference == tested_str


def test_too_many_indices():
    expr = 1.0
    for char in string.ascii_lowercase:
        idx_next = symbols(char)
        for i in range(1, 10):
            idx_previous = idx_next
            idx_next = symbols(f"{char}_{i}")
            expr *= DoubleVacuumTensorSymbol("X", (idx_previous,), (idx_next,))

    with pytest.raises(IndexError) as exec_info:
        generate_einsum(expr)

    assert exec_info.type == IndexError
    assert (
        exec_info.value.args[0]
        == "Too many indices!!! Not enough names for them."
    )


def test_pretty_indices():
    reference = """+np.einsum("Pp,qP,Qq,Rr,sR,Ss,aS,Aa,bA,Bb,pB->Qr", X_pp, X_qp, X_qq, X_rr, X_sr, X_ss, X_as, X_aa, X_ba, X_bb, X_pb)"""

    a, a_1 = symbols("a a_1", is_molA=True, above_fermi=True)
    i, i_1 = symbols("i i_1", is_molA=True, below_fermi=True)

    b, b_1 = symbols("b b_1", is_molB=True, above_fermi=True)
    j, j_1 = symbols("j j_1", is_molB=True, below_fermi=True)

    p, p_1 = symbols("p p_1", is_molA=True)
    q, q_1 = symbols("q q_1", is_molB=True)

    expr = (
        DoubleVacuumTensorSymbol("X", (a,), (a_1,))
        * DoubleVacuumTensorSymbol("X", (a_1,), (b,))
        * DoubleVacuumTensorSymbol("X", (b,), (b_1,))
        * DoubleVacuumTensorSymbol("X", (b_1,), (i,))
        * DoubleVacuumTensorSymbol("X", (i,), (i_1,))
        * DoubleVacuumTensorSymbol("X", (i_1,), (j,))
        * DoubleVacuumTensorSymbol("X", (j,), (j_1,))
        * DoubleVacuumTensorSymbol("X", (j_1,), (p,))
        * DoubleVacuumTensorSymbol("X", (p,), (p_1,))
        * DoubleVacuumTensorSymbol("X", (p_1,), (q,))
        * DoubleVacuumTensorSymbol("X", (q,), (q_1,))
    )
    tested_str = generate_einsum(expr, pretty_indices=True)

    assert reference == tested_str


def _matrix_product(coeff):
    """`coeff * A^{k}_{l} B^{l}_{m}`, the expression the coefficient
    formatting tests share"""
    k, l, m = symbols("k l m")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))

    return coeff * A * B


def test_integral_float_coefficient():
    # An integral Float is printed as an int, which relies on
    # `Float(2.0) == 2`. That holds up to sympy 1.12 and is False from
    # sympy 1.13 on, where the coefficient comes out as
    # "2.00000000000000" instead. This assertion is the tripwire for the
    # `sympy<1.13` cap in setup.py - see the version sensitivity note in
    # CLAUDE.md.
    reference = """+2 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(2.0))

    assert reference == tested_str


def test_negative_integral_float_coefficient():
    # a negative coefficient already carries its sign, no plus is prepended
    reference = """-2 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(-2.0))

    assert reference == tested_str


def test_non_integral_float_coefficient():
    reference = """+0.500000000000000 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(0.5))

    assert reference == tested_str


def test_rational_coefficient():
    reference = """+1/2 * np.einsum("kl,lm->km", A_kl, B_lm)"""

    tested_str = generate_einsum(_matrix_product(Rational(1, 2)))

    assert reference == tested_str


def test_sum_of_terms():
    # every term of an Add gets a line of its own
    reference = (
        '+np.einsum("kl,lm->km", A_kl, B_lm)\n' '+np.einsum("km->km", C_km)'
    )

    k, l, m = symbols("k l m")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))
    C = DoubleVacuumTensorSymbol("C", (m,), (k,))

    tested_str = generate_einsum(A * B + C)

    assert reference == tested_str


def test_pretty_indices_for_single_tensor():
    # the `pretty_indices` route for a bare tensor: `a_1` is renamed to
    # `r_1` by the psi4numpy mapping and then to `R` by the pretty table.
    # The variable name drops the subscript, exactly as the `Mul` route
    # does - see test_tensor_and_mul_routes_agree_on_variable_names.
    reference = """+np.einsum("Rsab->Rsab", t_rsab)"""

    a_1 = symbols("a_1", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    t = DoubleVacuumTensorSymbol("t", (i, j), (a_1, b))

    tested_str = generate_einsum(t, pretty_indices=True)

    assert reference == tested_str


def test_pretty_indices_rejects_index_outside_the_table():
    # `k` and `l` survive the psi4numpy mapping unchanged and are not in
    # `PRETTY_INDICES`, so the table cannot be applied
    k, l = symbols("k l")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (k,), (l,))

    with pytest.raises(ValueError) as exec_info:
        generate_einsum(A * B, pretty_indices=True)

    assert (
        exec_info.value.args[0]
        == "Code generator: `pretty_indices` cannot be applied!"
    )


@pytest.mark.xfail(
    strict=True,
    reason="a factor `generate_einsum` cannot translate is dropped "
    "silently: the fallback returns an empty string, so the term "
    "disappears from the generated code (leaving a blank line) instead of "
    "raising. The exception type asserted below is a proposal - whoever "
    "fixes this picks it",
)
def test_unsupported_term_is_not_dropped_silently():
    # `x` carries no tensor, so `generate_einsum` returns "" for it and the
    # term vanishes from the sum - the generated code is then quietly wrong
    k, l, m, x = symbols("k l m x")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))

    with pytest.raises((TypeError, ValueError)):
        generate_einsum(A * B + x)


def test_tensor_and_mul_routes_agree_on_variable_names():
    """A lone tensor and the same tensor inside a product have to name
    their array identically. The two routes build the name separately, so
    the subscript stripping has to be kept in step in both of them."""
    a_1 = symbols("a_1", is_molA=True, above_fermi=True)
    i_1 = symbols("i_1", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    t = DoubleVacuumTensorSymbol("t", (i_1, j), (a_1, b))

    bare = generate_einsum(t)
    in_a_product = generate_einsum(2.0 * t)

    assert '+np.einsum("csdb->csdb", t_rsab)' == bare
    assert '+2 * np.einsum("csdb->csdb", t_rsab)' == in_a_product
