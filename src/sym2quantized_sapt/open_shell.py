"""Unrestricted (open-shell) references: spin tags on the indices.

``is_alpha`` / ``is_beta`` on an index is the open-shell analogue of
the ``is_molA`` / ``is_molB`` monomer tag.  The core consults the two
predicates here: :func:`double_fermi_vac.contraction_double_vac` so
that a contraction across opposite spins vanishes, and
:func:`double_fermi_vac.evaluate_deltas_double_vac` so that a delta
does, and its surviving index keeps the tag.  Wick's theorem then does
the spin bookkeeping exactly, with no spin-summation rule at all.
Write each operator as its spin sectors and give each resolvent sector
its own normalisation.

The other open-shell route derives with spatial indices and sums over
spin per Goldstone loop afterwards:
:mod:`sym2quantized_sapt.spin_integrator.uhf`.  The two routes are
alternatives and share no code.
"""

__all__ = [
    "opposite_spins",
    "shared_spin_tag",
]


def shared_spin_tag(x, y) -> dict:
    """The spin assumptions an index standing in for ``x`` and ``y``
    must inherit.

    When two general (Fermi-level-free) indices contract, the fresh
    dummy projecting them onto the particle or hole space ranges over
    *their* spin, and so does the index that survives a delta between
    them: dropping the tag would silently sum both spins where the
    unrestricted blocks differ.  Untagged operands yield an empty
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
    analogue of the monomer tags: a contraction or a delta across them
    vanishes, which lets Wick's theorem do the UHF spin bookkeeping exactly
    instead of relying on the closed-shell ``2**loops`` rule."""
    ax, ay = x.assumptions0, y.assumptions0

    return bool(
        (ax.get("is_alpha") and ay.get("is_beta"))
        or (ax.get("is_beta") and ay.get("is_alpha"))
    )
