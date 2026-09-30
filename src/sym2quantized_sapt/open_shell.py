"""Unrestricted (open-shell) references: spin tags on the indices.

This is the supported UHF route.

``is_alpha`` / ``is_beta`` on an index is the open-shell analogue of
the ``is_molA`` / ``is_molB`` monomer tag.  The core consults the two
predicates here: :func:`double_fermi_vac.contraction_double_vac` so
that a contraction across opposite spins vanishes, and
:func:`double_fermi_vac.evaluate_deltas_double_vac` so that a delta
does, and its surviving index keeps the tag.  Wick's theorem then does
the spin bookkeeping exactly, with no spin-summation rule at all.
Write the perturbation as its spin sectors; the resolvent builder
:func:`sym2quantized_sapt.sapt_utils.get_R_nm` works unchanged, since
its untagged indices run over both spins and take the tags of the
indices they contract with.

Index renaming keeps the spins apart too, and :func:`index_latex`
prints a beta index with a bar over it.

A spin-tagged result is already resolved by spin, so the RHF
:func:`sym2quantized_sapt.spin_integrator.spin_integration` refuses it
(:func:`has_spin_tags`): it would count every loop a second time.

The per-loop route, :mod:`sym2quantized_sapt.spin_integrator.uhf`
(derive with spatial indices, then sum over spin per Goldstone loop),
is a frozen proof of concept, not maintained.  The two routes share no
code.
"""

from sympy import Symbol

__all__ = [
    "has_spin_tags",
    "index_latex",
    "index_spin",
    "opposite_spins",
    "shared_spin_tag",
]


def index_spin(index) -> str:
    """The spin letter of an index: ``"a"`` for ``is_alpha``, ``"b"`` for
    ``is_beta``, ``""`` for an untagged one.  These are the letters the
    generated array names and ``code_generator.array_table`` use."""
    assumptions = index.assumptions0

    if assumptions.get("is_alpha"):
        return "a"

    if assumptions.get("is_beta"):
        return "b"

    return ""


def has_spin_tags(expr) -> bool:
    """Whether any index of ``expr`` carries a spin tag (``is_alpha`` /
    ``is_beta``), i.e. whether ``expr`` belongs to the spin-tag route.
    One tagged index is enough; free (``Symbol``) and summed (``Dummy``)
    indices both count."""
    return any(index_spin(symbol) for symbol in expr.atoms(Symbol))


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


def index_latex(index) -> str:
    """The LaTeX of an index: its name, with a bar over a beta one.

    ``i_2`` tagged ``is_beta`` prints as ``\\bar{i}_2``; alpha and
    untagged indices print as their name.  The bar is added here, at
    print time, and never stored in ``index.name``: code generation
    identifies indices by name, so alpha and beta indices get distinct
    names from :func:`double_fermi_vac.substitute_dummies_double_vac`
    rather than the same name with different marks."""
    name = index.name

    if not index.assumptions0.get("is_beta"):
        return name

    base, separator, subscript = name.partition("_")

    return "\\bar{%s}%s%s" % (base, separator, subscript)
