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
becomes ``2**loops`` spin-blocked copies.  Loops are traced through
graph vertices only; a resolvent denominator is built with
``is_graph_vertex=False`` and takes its spins from the loops its
indices lie on.

:func:`rhf_collapse` is the consistency gate for the second route; read
its docstring for what that gate can and cannot prove.
"""

from itertools import product

from sympy import Add, Mul
from sympy.core import Expr
from sympy.physics.secondquant import TensorSymbol

from sym2quantized_sapt.spin_integrator import (
    _is_graph_vertex,
    _loop_partition,
)
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
#
# A tensor that is not a graph vertex (is_graph_vertex=False, the
# resolvent denominator) joins no lines, so its slot pairing means
# nothing and can disagree with the loops: e^{i j}_{a b} may have i and b on one loop.
# It is labelled per index instead, upper spins then lower spins --
# e_ab_ba[i, j, a, b] -- and keeps its spatial index layout.

#: default spin labels, alpha then beta
SPIN_LABELS = ("a", "b")
#: separates the tensor name from its block label, e.g. ``t_ab``
BLOCK_SEPARATOR = "_"


def _blocked(tensor, *label_groups):
    """``tensor`` renamed to its spin block, e.g. ``t`` -> ``t_ab``, or
    ``e`` -> ``e_ab_ba`` for a tensor that is not a graph vertex (one
    label group for the upper indices, one for the lower).

    Declared permutation symmetries are deliberately NOT carried over:
    a pair symmetry like ``t^{ij}_{ab} = t^{ji}_{ba}`` maps *between*
    blocks (``t_ab`` <-> ``t_ba``) for an unrestricted reference, and
    declaring it on one block would canonicalize within the block --
    silently wrong for the mixed-spin amplitudes.
    """
    name = BLOCK_SEPARATOR.join((str(tensor.symbol), *label_groups))

    return DoubleVacuumTensorSymbol(
        name,
        tuple(tensor.upper),
        tuple(tensor.lower),
        is_graph_vertex=_is_graph_vertex(tensor),
    )


def _split_block(tensor, labels=SPIN_LABELS):
    """``(base, upper_spins, lower_spins)`` of a spin-blocked tensor --
    one label per index, in slot order -- or ``None`` if ``tensor``
    carries no block label.  Inverse of :func:`_blocked`."""
    alphabet = set("".join(labels))
    name = str(tensor.symbol)
    n_upper, n_lower = len(tensor.upper), len(tensor.lower)

    if _is_graph_vertex(tensor):
        base, separator, suffix = name.rpartition(BLOCK_SEPARATOR)

        if (
            separator
            and n_upper == n_lower == len(suffix)
            and set(suffix) <= alphabet
        ):
            return base, suffix, suffix

        return None

    rest, separator, lower_spins = name.rpartition(BLOCK_SEPARATOR)
    base, separator_2, upper_spins = rest.rpartition(BLOCK_SEPARATOR)

    if (
        separator
        and separator_2
        and len(upper_spins) == n_upper
        and len(lower_spins) == n_lower
        and set(upper_spins + lower_spins) <= alphabet
    ):
        return base, upper_spins, lower_spins

    return None


def _spin_blocked_term(coefficients, tensors, labels):
    upper, lower = [], []

    for tensor in tensors:
        if not _is_graph_vertex(tensor):
            continue

        ups, lows = list(tensor.upper), list(tensor.lower)

        if len(ups) != len(lows):
            raise ValueError(
                f"spin blocking needs particle-conserving tensors; "
                f"{tensor} has {len(ups)} upper and {len(lows)} lower "
                f"indices."
            )

        upper += ups
        lower += lows

    on_lines = set(upper) | set(lower)
    for tensor in tensors:
        stray = [
            index
            for index in (*tensor.upper, *tensor.lower)
            if index not in on_lines
        ]

        if stray:
            raise ValueError(
                f"{tensor} is not a graph vertex (is_graph_vertex=False), "
                f"so its indices take their spin from the lines through "
                f"the vertices; {stray} lie on none."
            )

    loops = _loop_partition(upper, lower)

    blocked_terms = []
    for assignment in product(labels, repeat=len(loops)):
        spin_of_position = {}

        for loop, label in zip(loops, assignment):
            for position in loop:
                spin_of_position[position] = label

        # an index has the spin of the line it lies on
        spin_of_index = {}
        for position, label in spin_of_position.items():
            spin_of_index[upper[position]] = label
            spin_of_index[lower[position]] = label

        factors = list(coefficients)
        offset = 0
        for tensor in tensors:
            if not _is_graph_vertex(tensor):
                factors.append(
                    _blocked(
                        tensor,
                        "".join(spin_of_index[i] for i in tensor.upper),
                        "".join(spin_of_index[i] for i in tensor.lower),
                    )
                )
                continue

            n_pairs = len(tensor.upper)
            pair_labels = "".join(
                spin_of_position[offset + k] for k in range(n_pairs)
            )
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
    slot pair.  A tensor built with ``is_graph_vertex=False`` (the
    resolvent denominator) is not a vertex, so it is left out of the
    loops and labelled per index instead, upper then lower:
    ``e_ab_ba``.  Every one of its indices must lie on a line through
    the vertices; otherwise ``ValueError`` is raised.
    Indices in the returned expression refer to the spatial
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

    def strip(tensor):
        split = _split_block(tensor, labels)

        if split is None:
            return tensor

        return DoubleVacuumTensorSymbol(
            split[0],
            tuple(tensor.upper),
            tuple(tensor.lower),
            is_graph_vertex=_is_graph_vertex(tensor),
        )

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
