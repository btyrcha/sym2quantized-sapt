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

    assert tuple(swapped.upper) == tuple(plain.upper)
    assert tuple(swapped.lower) == tuple(plain.lower)
    assert swapped == plain


def test_no_symmetries_leaves_indices_untouched():
    """Without a symmetry set the given index order is preserved."""
    a_1, a_2, i_1, i_2 = _pair_indices()

    tensor = DoubleVacuumTensorSymbol("t", (i_2, i_1), (a_2, a_1))

    assert tuple(tensor.upper) == (i_2, i_1)
    assert tuple(tensor.lower) == (a_2, a_1)


def test_non_matching_symmetry_leaves_indices_untouched():
    """A symmetry set that cannot reach the sorted order is a no-op."""
    a_1, a_2, i_1, i_2 = _pair_indices()
    identity_only = (((0, 1), (0, 1)),)

    tensor = DoubleVacuumTensorSymbol(
        "t", (i_2, i_1), (a_2, a_1), identity_only
    )

    assert tuple(tensor.upper) == (i_2, i_1)
    assert tuple(tensor.lower) == (a_2, a_1)


def test_symmetries_roundtrip():
    """.symmetries returns what construction was given."""
    a_1, a_2, i_1, i_2 = _pair_indices()

    tensor = DoubleVacuumTensorSymbol(
        "t", (i_2, i_1), (a_2, a_1), FULL_SYMMETRIES
    )

    assert len(tensor.symmetries) == len(FULL_SYMMETRIES)


def test_no_symmetries_gives_empty_symmetry_tuple():
    """The default is an empty Tuple, not None."""
    a_1, _, i_1, _ = _pair_indices()

    tensor = DoubleVacuumTensorSymbol("t", (i_1,), (a_1,))

    assert len(tensor.symmetries) == 0


def test_dagger_swaps_indices_and_keeps_symmetries():
    """Dagger exchanges upper/lower and carries the symmetries over."""
    a_1, a_2, i_1, i_2 = _pair_indices()
    tensor = DoubleVacuumTensorSymbol(
        "t", (i_2, i_1), (a_2, a_1), FULL_SYMMETRIES
    )

    daggered = Dagger(tensor)

    assert tuple(daggered.upper) == tuple(tensor.lower)
    assert tuple(daggered.lower) == tuple(tensor.upper)
    assert len(daggered.symmetries) == len(FULL_SYMMETRIES)


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


def test_canonical_order_ignores_free_vs_summed_indices():
    """A free index must not move the canonical order.

    Basic.compare orders by class before content, so every free Symbol
    sorts ahead of every Dummy. Sorting on that would move the target
    order for a tensor mixing the two, no permutation in the symmetry
    group would match it, and the tensor would silently stay
    uncanonicalized - leaving `term + term` where `2*term` belongs.
    """
    a_1, a_2, i_1, _ = _pair_indices()
    i_2_free = symbols("i_2", is_molA=True, below_fermi=True)

    # a resolvent-like group: holes only permute with their particles
    paired = (((0, 1), (0, 1)), ((1, 0), (1, 0)))

    swapped = DoubleVacuumTensorSymbol(
        "e", (i_2_free, i_1), (a_2, a_1), paired
    )
    plain = DoubleVacuumTensorSymbol("e", (i_1, i_2_free), (a_1, a_2), paired)

    assert tuple(swapped.upper) == (i_1, i_2_free)
    assert tuple(swapped.lower) == (a_1, a_2)
    assert swapped == plain
    assert swapped + plain == 2 * plain


def test_canonicalization_survives_simultaneous_subs():
    """An index slot is not always a Symbol.

    subs(simultaneous=True) routes a Mul sentinel through the index
    slots while it resolves a swap, so the sort key must not assume
    every index has a .name.
    """
    a_1, a_2, i_1, i_2 = _pair_indices()
    paired = (((0, 1), (0, 1)), ((1, 0), (1, 0)))

    tensor = DoubleVacuumTensorSymbol("e", (i_1, i_2), (a_1, a_2), paired)

    # rename so the new names invert the order - the swap has to fire
    z = symbols("z", is_molA=True, below_fermi=True, cls=Dummy)
    b = symbols("b", is_molA=True, below_fermi=True, cls=Dummy)
    y = symbols("y", is_molA=True, above_fermi=True, cls=Dummy)
    c = symbols("c", is_molA=True, above_fermi=True, cls=Dummy)

    renamed = tensor.subs({i_1: z, i_2: b, a_1: y, a_2: c}, simultaneous=True)

    assert tuple(renamed.upper) == (b, z)
    assert tuple(renamed.lower) == (c, y)


def test_coupled_symmetry_canonicalizes_both_spellings():
    """A symmetry that couples upper and lower still canonicalizes.

    v^{pq}_{rs} = v^{qp}_{sr} cannot permute one row without the other,
    so the layout with both rows independently sorted is usually not in
    the orbit at all. Canonicalization has to pick the smallest layout
    the group can reach, or the two spellings stay distinct and their
    terms never collapse.
    """
    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    a_1 = symbols("a_1", is_molA=True, above_fermi=True, cls=Dummy)
    a_3 = symbols("a_3", is_molA=True, above_fermi=True)
    i_3 = symbols("i_3", is_molA=True, below_fermi=True)

    coupled = (((0, 1), (0, 1)), ((1, 0), (1, 0)))

    left = DoubleVacuumTensorSymbol("v", (a_1, i_3), (a_3, a), coupled)
    right = DoubleVacuumTensorSymbol("v", (i_3, a_1), (a, a_3), coupled)

    assert left == right
    assert left + right == 2 * left
