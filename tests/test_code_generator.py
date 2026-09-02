import pytest
import string

from sympy import symbols, Dummy
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
