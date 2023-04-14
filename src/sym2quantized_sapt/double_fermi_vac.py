from sympy.physics.secondquant import (
    AnnihilateFermion,
    CreateFermion,
    NO,
    contraction,
    _get_ordered_dummies,
)

from sympy import (
    S,
    Add,
    Mul,
    KroneckerDelta,
    Dummy,
    sympify,
)
from sympy import expand as sy_expand

from .operators import (
    AnnihilateFermion_A,
    AnnihilateFermion_B,
    CreateFermion_A,
    CreateFermion_B,
    DoubleFermiVaccum,
)


class NO_double_vac:
    """
    Normal ordering brackets for double Fermi vaccum.

    Assumes arg consits only is_molA, is_molB or commuting
    parts.
    """

    def __new__(cls, arg):
        arg = sympify(arg)
        arg = arg.expand()
        if isinstance(arg, Add):
            return Add(*[NO_double_vac(elem) for elem in arg.args])

        elif isinstance(arg, Mul):
            # separating coefficient from arg
            comuting_part, seq = arg.args_cnc()
            if comuting_part:
                coeff = Mul(*comuting_part)
                if not seq:
                    return coeff
            else:
                coeff = S.One

            # Some terms in the sequence may be already normaly ordered
            ordered = []
            part_A = []
            part_B = []
            for elem in seq:
                if isinstance(elem, NO):
                    ordered.append(elem)
                elif elem.is_molA:
                    part_A.append(elem)
                elif elem.is_molB:
                    part_B.append(elem)
                # elem neither is_molA, is_molB, NO nor commuting
                else:
                    print("Something went wrong:")
                    print(elem, "coresponds to neither molecule A nor B!")

            # apply normal ordering brackets to parts A and B separately
            return coeff * NO(Mul(*part_A)) * NO(Mul(*part_B)) * Mul(*ordered)

        # arg is neither Add nor Mul
        else:
            return arg


def contraction_double_vac(X, Y):
    """
    Calculates contraction for operators corresponding
    to either molecule A or molecule B.
    """

    if isinstance(X, DoubleFermiVaccum) and isinstance(Y, DoubleFermiVaccum):
        if isinstance(X, AnnihilateFermion_A) and isinstance(
            Y, CreateFermion_A
        ):
            if Y.state.assumptions0.get("below_fermi"):
                return S.Zero
            if X.state.assumptions0.get("below_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("a", is_molA=True, above_fermi=True)
            )

        if isinstance(X, CreateFermion_A) and isinstance(
            Y, AnnihilateFermion_A
        ):
            if Y.state.assumptions0.get("above_fermi"):
                return S.Zero
            if X.state.assumptions0.get("above_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("i", is_molA=True, below_fermi=True)
            )

        if isinstance(X, AnnihilateFermion_B) and isinstance(
            Y, CreateFermion_B
        ):
            if Y.state.assumptions0.get("below_fermi"):
                return S.Zero
            if X.state.assumptions0.get("below_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("b", is_molB=True, above_fermi=True)
            )

        if isinstance(X, CreateFermion_B) and isinstance(
            Y, AnnihilateFermion_B
        ):
            if Y.state.assumptions0.get("above_fermi"):
                return S.Zero
            if X.state.assumptions0.get("above_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("j", is_molB=True, below_fermi=True)
            )

        else:
            return S.Zero

    else:
        return contraction(X, Y)


def evaluate_deltas_double_vac(expr):
    """
    Function evaluating KroneckerDelta symbols in the expression assuming
    Einstein summation (the sum is over repeated index).

    Substiution are evaluated for double Fermi vaccum case. Therefore indicies
    in KronecerDelta should have an assumptions:
    - is_molA=True if this index applies only to part A of the complex,
    - is_molB=True if this index applies only to part B of the complex.
    """

    if isinstance(expr, Add):
        return Add(*[evaluate_deltas_double_vac(arg) for arg in expr.args])

    elif isinstance(expr, Mul):
        deltas = []
        indicies = {}
        for elem in expr.args:
            for s in elem.free_symbols:
                if s in indicies:
                    indicies[s] += 1
                else:
                    indicies[s] = 0
            if isinstance(elem, KroneckerDelta):
                deltas.append(elem)

        for d in deltas:
            # Now we have to check if killable and preferred apply
            # to the same part of the complex.
            killable_molA = d.killable_index.assumptions0.get("is_molA")
            killable_molB = d.killable_index.assumptions0.get("is_molB")
            preferred_molA = d.preferred_index.assumptions0.get("is_molA")
            preferred_molB = d.preferred_index.assumptions0.get("is_molB")

            if (killable_molA and preferred_molA) or (
                killable_molB and preferred_molB
            ):
                if d.killable_index.is_Symbol and indicies[d.killable_index]:
                    # Method killabel_index returns index containing less information
                    # regarding fermi level. If both contain the same amount of information
                    # alphabetical order is used to determine wich is preferred.
                    expr = expr.subs(d.killable_index, d.preferred_index)
                    if len(deltas) > 1:
                        return evaluate_deltas_double_vac(expr)

                elif (
                    d.preferred_index.is_Symbol
                    and indicies[d.preferred_index]
                    and d.indices_contain_equal_information
                ):
                    # Here we have situation where the preferred_index appers somewhere
                    # else in the expression. We can change
                    expr = expr.subs(d.preferred_index, d.killable_index)
                    if len(deltas) > 1:
                        return evaluate_deltas_double_vac(expr)

                else:
                    pass
            # Indicies correspond to diffrent parts of the complex! Delta is zero.
            else:
                return S.Zero

        return expr
    # Not Mul nor Add.
    else:
        return expr


def _get_contractions_double_vac(string, keep_only_fully_contracted=False):
    """
    Finds all posible contractions for given
    touple of fermionic operators. Returns Add object.

    Helper function.
    """

    if keep_only_fully_contracted:
        result = []
    else:
        result = [NO_double_vac(Mul(*string))]

    for i in range(len(string) - 1):
        for j in range(i + 1, len(string)):
            c = contraction_double_vac(string[i], string[j])

            if c:
                signchange = (j - i + 1) % 2
                if signchange:
                    coeff = S.NegativeOne * c
                else:
                    coeff = c

                # Operators posible for next level recursion
                string_next_level = string[i + 1 : j] + string[j + 1 :]

                if string_next_level:
                    result.append(
                        coeff
                        * NO_double_vac(
                            Mul(*string[:i])
                            * _get_contractions_double_vac(
                                string_next_level,
                                keep_only_fully_contracted=keep_only_fully_contracted,
                            )
                        )
                    )

                else:
                    result.append(coeff * NO_double_vac(Mul(*string[:i])))

        if keep_only_fully_contracted:
            break

    return Add(*result)


def wicks_double_vac(
    expr,
    expand=True,
    keep_only_fully_contracted=False,
    substitute_dummies=True,
    simplify_kronecker_deltas=True,
):
    """
    Returns Wicks Theorem expansion of a given expresion.
    """

    expr = sy_expand(expr)

    if isinstance(expr, Add):
        ans = Add(
            *[
                wicks_double_vac(
                    arg,
                    expand=expand,
                    keep_only_fully_contracted=keep_only_fully_contracted,
                    substitute_dummies=False,
                    simplify_kronecker_deltas=simplify_kronecker_deltas,
                )
                for arg in expr.args
            ]
        )
        if substitute_dummies:
            ans = substitute_dummies_double_vac(ans)
        return ans

    elif isinstance(expr, Mul):
        # Commuting and not-commuting parts
        com_part = []
        NO_part = []
        string = []
        for elem in expr.args:
            if elem.is_commutative:
                com_part.append(elem)
            elif isinstance(elem, NO):
                NO_part.append(elem)
            else:
                string.append(elem)

        string = tuple(string)
        n = len(string)

        if n == 0:
            ans = expr

        elif n == 1:
            if keep_only_fully_contracted:
                return S.Zero
            else:
                ans = expr

        else:
            # Finding all contractions
            ans = _get_contractions_double_vac(
                string,
                keep_only_fully_contracted=keep_only_fully_contracted,
            )
            ans = Mul(*com_part, *NO_part) * ans

        if expand:
            ans = ans.expand()
        if simplify_kronecker_deltas:
            ans = evaluate_deltas_double_vac(ans)
        if substitute_dummies:
            ans = substitute_dummies_double_vac(ans)

        return ans

    else:
        return expr


def get_fully_contracted(expr):
    """
    Leaves only fully contracted terms in a normal ordered expression.
    """

    expr = sy_expand(expr)

    if isinstance(expr, Add):
        return Add(*[get_fully_contracted(term) for term in expr.args])

    elif isinstance(expr, Mul):
        is_contr = True
        for term in expr.args:
            if (
                isinstance(term, NO)
                or isinstance(term, AnnihilateFermion)
                or isinstance(term, CreateFermion)
            ):
                is_contr = False
                break

        if is_contr:
            return expr
        else:
            return S.Zero

    else:
        return expr


def substitute_dummies_double_vac(expr):
    """
    Substiutute Dummy indicies systematicaly across the expresion
    making it posible to idenify terms that only differ due to index names.

    Always creates new Dummy indicies with the following manner:
    a, a_1, a_2, ... - particle indicies of molecule A,
    b, b_1, b_2, ... - particle indicies of molecule B,
    i, i_1, i_2, ... - hole indicies of molecule A,
    j, j_1, j_2, ... - hole indicies of molecule B,
    p, p_1, p_2, ... - general idnicies of molecule A,
    q, q_1, q_2, ... - general idnicies of molecule B.
    """

    molA_aboves = []
    molA_belows = []
    molA_generals = []

    molB_aboves = []
    molB_belows = []
    molB_generals = []

    dummies = expr.atoms(Dummy)
    a = b = i = j = p = q = 0
    for elem in dummies:
        assum = elem.assumptions0

        if assum.get("is_molA"):
            if assum.get("above_fermi"):
                if a == 0:
                    index = Dummy("a", **assum)
                else:
                    index = Dummy("a_{0}".format(a), **assum)
                molA_aboves.append(index)
                a += 1

            elif assum.get("below_fermi"):
                if i == 0:
                    index = Dummy("i", **assum)
                else:
                    index = Dummy("i_{0}".format(i), **assum)
                molA_belows.append(index)
                i += 1

            else:
                if p == 0:
                    index = Dummy("p", **assum)
                else:
                    index = Dummy("p_{0}".format(p), **assum)
                molA_generals.append(index)
                p += 1

        if assum.get("is_molB"):
            if assum.get("above_fermi"):
                if b == 0:
                    index = Dummy("b", **assum)
                else:
                    index = Dummy("b_{0}".format(b), **assum)
                molB_aboves.append(index)
                b += 1

            elif assum.get("below_fermi"):
                if j == 0:
                    index = Dummy("j", **assum)
                else:
                    index = Dummy("j_{0}".format(j), **assum)
                molB_belows.append(index)
                j += 1

            else:
                if q == 0:
                    index = Dummy("q", **assum)
                else:
                    index = Dummy("q_{0}".format(q), **assum)
                molB_generals.append(index)
                q += 1

    expr = expr.expand()
    terms = Add.make_args(expr)
    new_terms = []
    for term in terms:
        ordered = _get_ordered_dummies(term)

        a = iter(molA_aboves)
        i = iter(molA_belows)
        p = iter(molA_generals)

        b = iter(molB_aboves)
        j = iter(molB_belows)
        q = iter(molB_generals)

        subsdict = {}
        for d in ordered:
            assum = d.assumptions0

            if assum.get("is_molA"):
                if assum.get("above_fermi"):
                    subsdict[d] = next(a)
                elif assum.get("below_fermi"):
                    subsdict[d] = next(i)
                else:
                    subsdict[d] = next(p)

            if assum.get("is_molB"):
                if assum.get("above_fermi"):
                    subsdict[d] = next(b)
                elif assum.get("below_fermi"):
                    subsdict[d] = next(j)
                else:
                    subsdict[d] = next(q)

        subslist = []
        final_subs = []
        for k, v in subsdict.items():
            if k == v:
                continue
            if v in subsdict:
                # We check if the sequence of substitutions end quickly.  In
                # that case, we can avoid temporary symbols if we ensure the
                # correct substitution order.
                if subsdict[v] in subsdict:
                    # (x, y) -> (y, x),  we need a temporary variable
                    x = Dummy("x")
                    subslist.append((k, x))
                    final_subs.append((x, v))
                else:
                    # (x, y) -> (y, a),  x->y must be done last
                    # but before temporary variables are resolved
                    final_subs.insert(0, (k, v))
            else:
                subslist.append((k, v))
        subslist.extend(final_subs)
        new_terms.append(term.subs(subslist))

    return Add(*new_terms)


def commutator(A, B):
    """
    Commutator of two operators A and B.
    """

    comm = A * B - B * A
    comm = sy_expand(comm)

    return comm
