# find_theta.py — robust principal-root solver for Grover angle θ
# Qiskit 2.x / Python 3.9+

import math

# Chebyshev U_n(x) via stable 3-term recurrence
def _U(n: int, x: float) -> float:
    if n == 0: 
        return 1.0
    if n == 1: 
        return 2.0 * x
    U_nm2, U_nm1 = 1.0, 2.0 * x
    for _ in range(2, n + 1):
        U_n = 2.0 * x * U_nm1 - U_nm2
        U_nm2, U_nm1 = U_nm1, U_n
    return U_nm1

def _f_x(x: float, m: int, Pm: float) -> float:
    """
    Rewrite Pm = 1/2 − sin(4mθ)/(4m·sin(2θ))
    using sin(nφ) = sin φ · U_{n−1}(cos φ) with φ=2θ, x=cos(2θ):
    sin(4mθ)/sin(2θ) = U_{4m−1}(cos(2θ)) = U_{4m−1}(x).
    So:  0 = 1/2 − Pm − (1/(4m)) U_{4m−1}(x).
    """
    return 0.5 - Pm - _U(4*m - 1, x) / (4.0 * m)

def find_theta(m: int, Pm: float, tol: float = 1e-12, maxit: int = 80) -> float:
    """
    Return the *principal* Grover angle θ ∈ (0, π/(4m))
    that solves: Pm = 1/2 − sin(4mθ)/(4m·sin(2θ)).

    Strategy: bisection on x = cos(2θ) over x ∈ (cos(π/(2m)), 1).
    On this interval the principal branch is monotone, yielding a single root.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    if not (0.0 <= Pm <= 1.0):
        raise ValueError("Pm must be in [0,1]")

    # principal interval mapping:
    # θ ∈ (0, π/(4m))  ↔  2θ ∈ (0, π/(2m))  ↔  x=cos(2θ) ∈ (cos(π/(2m)), 1)
    eps = 1e-15
    xL = math.cos(math.pi / (2.0 * m)) + eps
    xR = 1.0 - eps

    fL = _f_x(xL, m, Pm)
    fR = _f_x(xR, m, Pm)

    # If interval doesn't straddle, gently widen left endpoint (rare edge cases)
    if fL * fR > 0:
        xL = max(0.0 + 1e-12, math.cos(math.pi / (2.0 * m + 2.0)))
        fL = _f_x(xL, m, Pm)
        if fL * fR > 0:
            # As a final guard, fall back to near-global left bound
            xL = 0.0 + 1e-12
            fL = _f_x(xL, m, Pm)
            if fL * fR > 0:
                raise ValueError(f"Could not bracket θ for m={m}, Pm={Pm}")

    # Bisection
    for _ in range(maxit):
        xm = 0.5 * (xL + xR)
        fm = _f_x(xm, m, Pm)
        if abs(fm) < tol or abs(xR - xL) < tol:
            x = xm
            break
        if fL * fm <= 0:
            xR, fR = xm, fm
        else:
            xL, fL = xm, fm
    else:
        x = 0.5 * (xL + xR)

    theta = 0.5 * math.acos(x)  # θ = ½ arccos(cos(2θ))
    # clamp to principal interval just in case of numeric drift
    theta = min(max(theta, 0.0 + 1e-14), math.pi/(4.0*m) - 1e-14)
    return theta

# (Optional) helper if you still want all roots over (0, π/2) for diagnostics.
def find_thetas_all(m: int, Pm: float, intervals: int = None):
    """
    Return all roots θ in (0, π/2); mainly for debugging.
    Uses coarse subdivision then bisection per bracket.
    """
    import numpy as np
    a, b = 0.0 + 1e-9, math.pi/2 - 1e-9
    # heuristic: ~10 samples per oscillation of sin(4mθ) ⇒ ~20m intervals
    intervals = intervals or max(200, 20*m)
    xs = np.cos(2*np.linspace(a, b, intervals+1))  # monotone map via x=cos(2θ)
    roots = []
    for xL, xR in zip(xs[:-1], xs[1:]):
        fL, fR = _f_x(xL, m, Pm), _f_x(xR, m, Pm)
        if np.isnan(fL) or np.isnan(fR) or np.isinf(fL) or np.isinf(fR):
            continue
        if abs(fL) < 1e-12:
            roots.append(0.5*math.acos(xL)); continue
        if abs(fR) < 1e-12:
            roots.append(0.5*math.acos(xR)); continue
        if fL * fR < 0:
            # bisection in x, then map back to θ
            for _ in range(60):
                xm = 0.5*(xL+xR); fm = _f_x(xm, m, Pm)
                if abs(fm) < 1e-12 or abs(xR-xL) < 1e-12: break
                if fL * fm <= 0: xR, fR = xm, fm
                else:            xL, fL = xm, fm
            roots.append(0.5*math.acos(0.5*(xL+xR)))
    # dedupe & sort
    roots = sorted({round(float(r), 12) for r in roots})
    return roots
