import pytest

from sympy import Dummy, S, latex, symbols

from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.double_fermi_vac import (
    NO_double_vac,
    get_fully_contracted,
    substitute_dummies_double_vac,
    CreateFermion_A,
    AnnihilateFermion_A,
)


def _one_particle_tensor():
    a = symbols("a", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)

    return DoubleVacuumTensorSymbol("f", (i,), (a,)), a, i


def test_get_fully_contracted_keeps_only_closed_terms():
    """a term still carrying operators - normal ordered or bare - is not
    fully contracted and is dropped"""
    reference = r"2 f^{i}_{a}"

    f, a, i = _one_particle_tensor()

    closed = 2 * f
    normal_ordered = f * NO_double_vac(
        CreateFermion_A(a) * AnnihilateFermion_A(i)
    )
    bare_operator = f * CreateFermion_A(a)

    tested_str = latex(
        get_fully_contracted(closed + normal_ordered + bare_operator)
    )

    assert reference == tested_str


def test_get_fully_contracted_drops_a_normal_ordered_term():
    f, a, i = _one_particle_tensor()

    normal_ordered = f * NO_double_vac(
        CreateFermion_A(a) * AnnihilateFermion_A(i)
    )

    assert get_fully_contracted(normal_ordered) == S.Zero


def test_get_fully_contracted_passes_a_bare_tensor_through():
    """neither an Add nor a Mul - the expression is returned unchanged"""
    f, _, _ = _one_particle_tensor()

    assert get_fully_contracted(f) == f


def test_dummy_ordering_uses_normal_ordered_operators():
    """
    The same operator written with its two particle indicies swapped has to
    canonicalize to the same expression.
    """
    a_1, a_2 = symbols("a_1 a_2", is_molA=True, above_fermi=True, cls=Dummy)

    one_way = NO_double_vac(CreateFermion_A(a_1) * AnnihilateFermion_A(a_2))
    other_way = NO_double_vac(CreateFermion_A(a_2) * AnnihilateFermion_A(a_1))

    assert latex(substitute_dummies_double_vac(one_way)) == latex(
        substitute_dummies_double_vac(other_way)
    )


@pytest.mark.skip(
    reason="TEMPLATE - no expression is known to reach the branch; see "
    "TODO.md, `substitute_dummies_double_vac` substitution cycles"
)
def test_substitution_cycle_needs_a_temporary_symbol():
    """
    Covers the `(x, y) -> (y, x)` branch of `substitute_dummies_double_vac`
    (`double_fermi_vac.py:712-724`), which swaps two dummies through a
    temporary symbol.

    The branch is guarded by `if v in subsdict`, that is: a replacement
    dummy has to be one of the dummies being replaced. The replacements are
    built fresh with `Dummy(...)` inside every call and `Dummy` equality
    includes a hidden index, so the guard looks unreachable by
    construction - the branch is inherited from sympy's
    `substitute_dummies`, where the replacement pool is fixed rather than
    freshly created.

    Before writing this test, settle that question first: either build an
    expression that reaches the branch (and assert the swap comes out
    right), or delete the branch and its `final_subs` bookkeeping.
    """
    raise NotImplementedError
