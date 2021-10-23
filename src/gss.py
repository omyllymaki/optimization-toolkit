import math

INV_PHI = (math.sqrt(5) - 1) / 2  # 1 / phi
INV_PHI_SUARE = (3 - math.sqrt(5)) / 2  # 1 / phi^2


def gss(f, a, b, max_iter=10, tol=None):
    """
    Golden-section search.

    Given a function f with a single local minimum in the interval [a,b], gss returns a subset interval [c,d] that
    contains the minimum.
    """

    a, b = min(a, b), max(a, b)
    h = b - a
    if tol is not None:
        if h <= tol:
            return a, b

    # Required steps to achieve tolerance
    if tol is None:
        n_iter = max_iter
    else:
        n_iter = min(int(math.ceil(math.log(tol / h) / math.log(INV_PHI))), max_iter)

    c = a + INV_PHI_SUARE * h
    d = a + INV_PHI * h
    yc = f(c)
    yd = f(d)

    for k in range(n_iter - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = INV_PHI * h
            c = a + INV_PHI_SUARE * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = INV_PHI * h
            d = a + INV_PHI * h
            yd = f(d)

    if yc < yd:
        return a, d
    else:
        return c, b
