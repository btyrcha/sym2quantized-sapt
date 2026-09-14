"""Unrestricted (open-shell) references.

Two independent mechanisms live here, and choosing between them is the
one decision this module asks of a caller.

**Spin tags** (:func:`opposite_spins`, :func:`shared_spin_tag`) are the
general route: ``is_alpha`` / ``is_beta`` on an index is the open-shell
analogue of the ``is_molA`` / ``is_molB`` monomer tag, and
:func:`double_fermi_vac.contraction_double_vac` consults them so that a
contraction across opposite spins vanishes.  Wick's theorem then does
the spin bookkeeping exactly, with no spin-summation rule at all.
Write each operator as its spin sectors and give each resolvent sector
its own normalisation.

**Per-loop summation** (:func:`spin_integration_uhf`) is the cheap
route: spin is constant along a Goldstone loop, so a spatial term
becomes ``2**loops`` spin-blocked copies.  It is valid only where no
projector carries two or more index pairs in one space -- see the
warning on the function, which a benchmark had to teach us.

:func:`rhf_collapse` is the consistency gate for the second route; read
its docstring for what that gate can and cannot prove.
"""

from itertools import product

from sympy import Add, Mul
from sympy.core import Expr
from sympy.physics.secondquant import TensorSymbol

from sym2quantized_sapt.spin_integrator import _loop_partition
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol

__all__ = [
    "BLOCK_SEPARATOR",
    "SPIN_LABELS",
    "opposite_spins",
    "rhf_collapse",
    "shared_spin_tag",
    "spin_integration_uhf",
]


# --- spin tags: the Wick-level mechanism ------------------------------


def shared_spin_tag(x, y) -> dict:
    """The spin assumptions a fallback summation dummy must inherit.

    When two general (Fermi-level-free) indices contract, the fresh
    dummy projecting them onto the particle or hole space ranges over
    *their* spin: dropping the tag would silently sum both spins where
    the unrestricted blocks differ.  Untagged operands yield an empty
    dict, keeping the closed-shell behaviour."""
    for assumptions in (x.assumptions0, y.assumptions0):
        if assumptions.get("is_alpha"):
            return {"is_alpha": True}
        if assumptions.get("is_beta"):
            return {"is_beta": True}
    return {}


def opposite_spins(x, y) -> bool:
    """Both indices carry an explicit spin tag (``is_alpha`` /
    ``is_beta``) and the tags differ.  Spin tags are the open-shell
    analogue of the monomer tags: a contraction across them vanishes,
    which lets Wick's theorem do the UHF spin bookkeeping exactly
    instead of relying on the closed-shell ``2**loops`` rule."""
    ax, ay = x.assumptions0, y.assumptions0
    return bool(
        (ax.get("is_alpha") and ay.get("is_beta"))
        or (ax.get("is_beta") and ay.get("is_alpha"))
    )


# --- per-loop spin blocking -------------------------------------------
#
# For an unrestricted reference the two spin cases of a loop are no
# longer identical, so the RHF factor 2**loops becomes an explicit sum:
# every term turns into 2**loops spin-labelled copies, with each tensor
# replaced by its spin *block*.  A block is labelled per slot pair --
# every (upper_k, lower_k) pair of a particle-conserving tensor carries
# one spin -- which is exactly the "pair of indices belongs to one
# spin" structure of unrestricted arrays: t_ab[i, j, a, b] has the
# (i, a) line alpha and the (j, b) line beta, v_ba likewise, and the
# spatial index ranges differ per spin.

#: default spin labels, alpha then beta
SPIN_LABELS = ("a", "b")
#: separates the tensor name from its block label, e.g. ``t_ab``
BLOCK_SEPARATOR = "_"


def _blocked(tensor, pair_labels):
    """``tensor`` renamed to its spin block, e.g. ``t`` -> ``t_ab``.

    Declared permutation symmetries are deliberately NOT carried over:
    a pair symmetry like ``t^{ij}_{ab} = t^{ji}_{ba}`` maps *between*
    blocks (``t_ab`` <-> ``t_ba``) for an unrestricted reference, and
    declaring it on one block would canonicalize within the block --
    silently wrong for the mixed-spin amplitudes.
    """
    name = str(tensor.symbol) + BLOCK_SEPARATOR + "".join(pair_labels)
    return DoubleVacuumTensorSymbol(
        name, tuple(tensor.upper), tuple(tensor.lower)
    )


def _spin_blocked_term(coefficients, tensors, labels):
    upper, lower, pair_owner = [], [], []
    for tensor_index, tensor in enumerate(tensors):
        ups, lows = list(tensor.upper), list(tensor.lower)
        if len(ups) != len(lows):
            raise ValueError(
                f"spin blocking needs particle-conserving tensors; "
                f"{tensor} has {len(ups)} upper and {len(lows)} lower "
                f"indices."
            )
        upper += ups
        lower += lows
        pair_owner += [tensor_index] * len(ups)

    loops = _loop_partition(upper, lower)

    blocked_terms = []
    for assignment in product(labels, repeat=len(loops)):
        spin_of_position = {}
        for loop, label in zip(loops, assignment):
            for position in loop:
                spin_of_position[position] = label
        factors = list(coefficients)
        offset = 0
        for tensor in tensors:
            n_pairs = len(tensor.upper)
            pair_labels = [
                spin_of_position[offset + k] for k in range(n_pairs)
            ]
            factors.append(_blocked(tensor, pair_labels))
            offset += n_pairs
        blocked_terms.append(Mul(*factors))
    return Add(*blocked_terms)


def spin_integration_uhf(expr: Expr, labels=SPIN_LABELS) -> Expr:
    """Spin summation for an Unrestricted Hartree-Fock reference.

    Where :func:`spin_integration` multiplies each term by
    ``2**loops``, here each term becomes the explicit sum over one spin
    label per Goldstone loop, with every tensor replaced by its spin
    block: ``t`` becomes ``t_aa``, ``t_ab``, ... with one label per
    slot pair.  Indices in the returned expression refer to the spatial
    orbitals *of that spin* -- the alpha and beta index ranges of an
    unrestricted reference differ, which is why the blocks are distinct
    arrays rather than views of one.

    Open (external) lines are enumerated like closed ones, so an
    operator-valued expression comes back with one term per spin block
    of the result; select the block by the external labels if only one
    is wanted.

    Setting all blocks of every tensor equal must reproduce
    :func:`spin_integration` term by term -- the RHF collapse; see
    :func:`rhf_collapse` for the symbolic form of that gate.

    .. warning::

       The per-loop route is only as correct as the spatial expression
       it is applied to, and the spatial-Wick + ``2**loops`` pipeline
       is **not** valid for expressions containing a resolvent (or any
       projector) with two or more index pairs in one space: part of
       the projector's permutation multiplicity flows through
       exchange-wired contractions, which spatial terms can only carry
       with same-spin labels.  Concretely, ``<W R_(2,0) W>`` treated
       this way halves the opposite-spin MP2 energy (its closed-shell
       limit is ``1.5A - B`` instead of ``2A - B``) -- caught by the
       psi4 benchmark, see ``docs/notes/uhf-spin-summation.md``.  For
       MP-n and any multi-pair projector, tag the indices with
       ``is_alpha`` / ``is_beta`` instead and let Wick's theorem do
       the spin bookkeeping (the contraction rule vanishes across
       spin tags), with per-sector resolvent normalisation --
       ``1/(n!)**2`` per same-spin pair group, distinguishable pairs
       unpermuted.  Single-pair-per-space projections (``R_(1,1)``
       dispersion, the eq 46 dressings) are unaffected.
    """
    if isinstance(expr, Add):
        return Add(*[spin_integration_uhf(arg, labels) for arg in expr.args])

    if isinstance(expr, Mul):
        coefficients, tensors = [], []
        for elem in expr.args:
            if isinstance(elem, TensorSymbol):
                tensors.append(elem)
            else:
                coefficients.append(elem)
        return _spin_blocked_term(coefficients, tensors, labels)

    if isinstance(expr, TensorSymbol):
        return _spin_blocked_term([], [expr], labels)

    return expr


def rhf_collapse(expr: Expr, labels=SPIN_LABELS) -> Expr:
    """Strip the spin-block labels off every tensor of a
    :func:`spin_integration_uhf` result.

    With all blocks of a tensor set equal -- which is what a restricted
    reference means -- the ``2**loops`` copies of each term become
    identical and sympy sums them back up, so
    ``rhf_collapse(spin_integration_uhf(e)) == spin_integration(e)``
    exactly.  That identity is the machine-checkable gate every UHF
    derivation should pass before its blocks are trusted.
    """
    alphabet = set("".join(labels))

    def strip(tensor):
        name = str(tensor.symbol)
        base, separator, suffix = name.rpartition(BLOCK_SEPARATOR)
        n_pairs = len(tensor.upper)
        if separator and len(suffix) == n_pairs and set(suffix) <= alphabet:
            return DoubleVacuumTensorSymbol(
                base, tuple(tensor.upper), tuple(tensor.lower)
            )
        return tensor

    if isinstance(expr, Add):
        return Add(*[rhf_collapse(arg, labels) for arg in expr.args])
    if isinstance(expr, Mul):
        return Mul(
            *[
                strip(elem) if isinstance(elem, TensorSymbol) else elem
                for elem in expr.args
            ]
        )
    if isinstance(expr, TensorSymbol):
        return strip(expr)
    return expr
