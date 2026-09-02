import pytest

from sympy import Dummy, S, symbols

from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def _mol_indices():
    """fresh (a, i, b, j) dummies: particle/hole on monomer A and B"""
    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)
    b = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
    j = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)
    return a, i, b, j


def _linked_term():
    """v^{ab}_{ij} t^{ij}_{ab} - every index shared, one component"""
    a, i, b, j = _mol_indices()
    return DoubleVacuumTensorSymbol(
        "v", (a, b), (i, j)
    ) * DoubleVacuumTensorSymbol("t", (i, j), (a, b))


def _unlinked_term():
    """o_B^{a}_{i} o_A^{b}_{j} - no shared index, two components"""
    a, i, b, j = _mol_indices()
    return DoubleVacuumTensorSymbol(
        "o_B", (a,), (i,)
    ) * DoubleVacuumTensorSymbol("o_A", (b,), (j,))


def test_keeps_linked_term():
    """A connected term survives untouched."""
    term = _linked_term()

    assert get_only_linked(term) == term


def test_drops_unlinked_term():
    """A term whose tensors share no index is discarded."""
    assert get_only_linked(_unlinked_term()) == S.Zero


def test_add_keeps_only_linked_terms():
    """Across a sum, only the connected terms are kept."""
    linked = _linked_term()
    expr = linked + _unlinked_term()

    assert get_only_linked(expr) == linked


def test_passes_through_bare_tensor():
    """A lone TensorSymbol is neither Add nor Mul and is returned as-is."""
    a, i, _, _ = _mol_indices()
    tensor = DoubleVacuumTensorSymbol("o_B", (a,), (i,))

    assert get_only_linked(tensor) == tensor


def test_scalar_term_without_tensors():
    """A coefficient-only Mul has no diagram, and must not crash."""
    x = symbols("x")
    expr = 2 * x

    assert get_only_linked(expr) == expr


def test_no_indices_tensor():
    """A tensor withour no indicies is a scalar (coefficient only)."""
    tensor = DoubleVacuumTensorSymbol("V_0", (), ())

    assert get_only_linked(tensor) == tensor
