import math

def theta_from_Pm(m, Pm, method="bisection", tol=1e-12, maxit=200):
    def f(th):
        return 0.5 - (math.sin(4*m*th) / (4*m*math.sin(2*th))) - Pm

    # Bracket a root with a coarse scan
    K = 2000
    a, fa = 1e-12, f(1e-12)
    root_interval = None
    for k in range(1, K+1):
        b = (math.pi/2 - 1e-12) * k / K
        fb = f(b)
        if fa*fb <= 0:
            root_interval = (a, b)
            break
        a, fa = b, fb
    if root_interval is None:
        raise RuntimeError("Failed to bracket a root on (0, pi/2).")

    a, b = root_interval

    if method == "bisection":
        for _ in range(maxit):
            c = 0.5*(a+b)
            fc = f(c)
            if abs(fc) < tol or (b-a) < tol:
                return c
            if f(a)*fc <= 0:
                b = c
            else:
                a = c
        return 0.5*(a+b)

    else:  # hybrid: a few Newton steps from midpoint, with fallbacks
        th = 0.5*(a+b)
        for _ in range(maxit):
            s2 = math.sin(2*th)
            if abs(s2) < 1e-14:  # avoid division issues
                th = 0.5*(a+b)
            # f and f' (from quotient rule)
            g = math.sin(4*m*th)
            h = 4*m*s2
            fp = -((4*m*math.cos(4*m*th))*h - g*(8*m*math.cos(2*th))) / (h*h)
            val = 0.5 - g/h - Pm
            if abs(val) < tol:
                return th
            step = val/fp if fp != 0 else 0.0
            th_new = th - step
            # keep iterate inside (a,b); if it escapes, bisect once
            if not (a < th_new < b) or math.isnan(th_new):
                th_new = 0.5*(a+b)
            # update bracket using sign
            if f(a)*f(th_new) <= 0:
                b = th_new
            else:
                a = th_new
            th = th_new
        return th
