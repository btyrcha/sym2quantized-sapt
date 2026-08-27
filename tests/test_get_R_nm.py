from sympy import Add, Mul, S, latex
from sympy.physics.secondquant import TensorSymbol

from sym2quantized_sapt.sapt_utils import get_R_nm, get_V_operator


def _tensors_of(term):
    """the TensorSymbol factors of a single term"""
    factors = term.args if isinstance(term, Mul) else (term,)
    return [factor for factor in factors if isinstance(factor, TensorSymbol)]


def test_R_10_expands_to_two_terms():
    """V contributes v^{ij}_{aj} and (v_B)^{i}_{a} to a single A excitation."""
    result = get_R_nm(1, 0, get_V_operator())

    assert isinstance(result, Add)
    assert len(result.args) == 2


def test_R_01_expands_to_two_terms():
    """The monomer-B mirror of test_R_10_expands_to_two_terms."""
    result = get_R_nm(0, 1, get_V_operator())

    assert isinstance(result, Add)
    assert len(result.args) == 2


def test_R_11_is_a_single_term():
    """Only the two-electron part of V survives a simultaneous A and B
    excitation, so R(1,1) collapses to one term."""
    result = get_R_nm(1, 1, get_V_operator())

    assert isinstance(result, Mul)


def test_R_11_carries_dispersion_index_structure():
    """e and v share (i, j) upper / (a, b) lower - the E(20)_disp shape."""
    result = get_R_nm(1, 1, get_V_operator())

    by_symbol = {str(t.symbol()): t for t in _tensors_of(result)}

    assert set(by_symbol) == {"e", "v"}
    assert by_symbol["e"].upper() == by_symbol["v"].upper()
    assert by_symbol["e"].lower() == by_symbol["v"].lower()


def test_R_20_of_V_vanishes():
    """V acts on one electron per monomer, so it cannot doubly excite A."""
    assert get_R_nm(2, 0, get_V_operator()) == S.Zero


R_V_10_REFERENCE = (
    r"1.0 e^{i}_{a} v^{ij}_{aj} a^\dagger_{a} a_{i} "
    r"+ 1.0 e^{i}_{a} v_B^{i}_{a} a^\dagger_{a} a_{i}"
)

R_V_01_REFERENCE = (
    r"1.0 e^{j}_{b} v^{ij}_{ib} b^\dagger_{b} b_{j} "
    r"+ 1.0 e^{j}_{b} v_A^{j}_{b} b^\dagger_{b} b_{j}"
)

R_V_11_REFERENCE = (
    r"1.0 e^{ij}_{ab} v^{ij}_{ab} a^\dagger_{a} a_{i} b^\dagger_{b} b_{j}"
)


def test_R_11_matches_reference():
    """Exact expression for the E(20)_disp resolvent."""
    result = get_R_nm(1, 1, get_V_operator())

    assert latex(result) == R_V_11_REFERENCE


def test_R_10_matches_reference():
    result = get_R_nm(1, 0, get_V_operator())

    assert latex(result) == R_V_10_REFERENCE


def test_R_01_matches_reference():
    result = get_R_nm(0, 1, get_V_operator())

    assert latex(result) == R_V_01_REFERENCE
