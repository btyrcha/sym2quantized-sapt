from sympy import symbols
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.code_generator import generate_einsum


def test_two_eris():

    reference = """-np.einsum("abrs,rsab", v_abrs, v_rsab)"""

    a = symbols("a", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)

    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)

    v_ijab = DoubleVacuumTensorSymbol("v", (a, b,), (i, j,))
    v_abij = DoubleVacuumTensorSymbol("v", (i, j,), (a, b,))

    tested_expr = generate_einsum((-1.0) * v_ijab * v_abij)

    assert reference == tested_expr


def test_matrix_multiplication():

    reference = """+np.einsum("kl,lm->km", A_kl, B_lm)"""

    k, l, m = symbols("k l m")

    A = DoubleVacuumTensorSymbol("A", (l,), (k,))
    B = DoubleVacuumTensorSymbol("B", (m,), (l,))

    tested_str = generate_einsum(A * B)

    assert reference == tested_str
