import itertools

import pytest

from sympy import Dummy, symbols
from sympy.physics.secondquant import Dagger

from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


# every (upper, lower) permutation pair for a rank-2 / rank-2 tensor
FULL_SYMMETRIES = tuple(itertools.product([(0, 1), (1, 0)], repeat=2))


def _pair_indices():
    """two particle and two hole dummies on monomer A"""
    a_1 = symbols("a_1", is_molA=True, above_fermi=True, cls=Dummy)
    a_2 = symbols("a_2", is_molA=True, above_fermi=True, cls=Dummy)
    i_1 = symbols("i_1", is_molA=True, below_fermi=True, cls=Dummy)
    i_2 = symbols("i_2", is_molA=True, below_fermi=True, cls=Dummy)
    return a_1, a_2, i_1, i_2


def test_symmetries_canonicalize_permuted_indices():
    """Swapped indices plus the matching symmetry set canonicalize.

    This is what lets equivalent terms collapse: two tensors written
    with their indices in different orders must compare equal.
    """
    a_1, a_2, i_1, i_2 = _pair_indices()

    plain = DoubleVacuumTensorSymbol("t", (i_1, i_2), (a_1, a_2))
    swapped = DoubleVacuumTensorSymbol(
        "t", (i_2, i_1), (a_2, a_1), FULL_SYMMETRIES
    )

    assert tuple(swapped.upper()) == tuple(plain.upper())
    assert tuple(swapped.lower()) == tuple(plain.lower())
    assert swapped == plain


def test_no_symmetries_leaves_indices_untouched():
    """Without a symmetry set the given index order is preserved."""
    a_1, a_2, i_1, i_2 = _pair_indices()

    tensor = DoubleVacuumTensorSymbol("t", (i_2, i_1), (a_2, a_1))

    assert tuple(tensor.upper()) == (i_2, i_1)
    assert tuple(tensor.lower()) == (a_2, a_1)


def test_non_matching_symmetry_leaves_indices_untouched():
    """A symmetry set that cannot reach the sorted order is a no-op."""
    a_1, a_2, i_1, i_2 = _pair_indices()
    identity_only = (((0, 1), (0, 1)),)

    tensor = DoubleVacuumTensorSymbol(
        "t", (i_2, i_1), (a_2, a_1), identity_only
    )

    assert tuple(tensor.upper()) == (i_2, i_1)
    assert tuple(tensor.lower()) == (a_2, a_1)


def test_get_symmetries_roundtrip():
    """get_symmetries() returns what construction was given."""
    a_1, a_2, i_1, i_2 = _pair_indices()

    tensor = DoubleVacuumTensorSymbol(
        "t", (i_2, i_1), (a_2, a_1), FULL_SYMMETRIES
    )

    assert len(tensor.get_symmetries()) == len(FULL_SYMMETRIES)


def test_no_symmetries_gives_empty_symmetry_tuple():
    """The default is an empty Tuple, not None."""
    a_1, _, i_1, _ = _pair_indices()

    tensor = DoubleVacuumTensorSymbol("t", (i_1,), (a_1,))

    assert len(tensor.get_symmetries()) == 0


def test_dagger_swaps_indices_and_keeps_symmetries():
    """Dagger exchanges upper/lower and carries the symmetries over."""
    a_1, a_2, i_1, i_2 = _pair_indices()
    tensor = DoubleVacuumTensorSymbol(
        "t", (i_2, i_1), (a_2, a_1), FULL_SYMMETRIES
    )

    daggered = Dagger(tensor)

    assert tuple(daggered.upper()) == tuple(tensor.lower())
    assert tuple(daggered.lower()) == tuple(tensor.upper())
    assert len(daggered.get_symmetries()) == len(FULL_SYMMETRIES)


def test_malformed_symmetry_entry():
    """A symmetry entry that is not an (upper, lower) pair."""
    a_1, a_2, i_1, i_2 = _pair_indices()
    malformed = (((0, 1),),)  # missing the lower-index permutation

    with pytest.raises(IndexError) as exec_info:
        DoubleVacuumTensorSymbol("t", (i_2, i_1), (a_2, a_1), malformed)

    assert exec_info.type == IndexError
    assert (
        exec_info.value.args[0]
        == f"Symmetry must be (upper, lower) permutation pair, was {malformed[0]}!"
    )
