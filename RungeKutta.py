import math
import numpy as np
from scipy import interpolate

##### f dde model
##### u initial value
##### h history function
##### ts time span
##### k  1/k is the fixed step size

def rk4(f, u, h, p, ts, k):
    tslot = np.linspace(ts[0], ts[1], round((ts[1] - ts[0]) * k))
    n  = len(tslot)
    uslot = np.zeros( (n, 2) )
    uslot[0] = u
    delta = tslot[1] - tslot[0]
    def hnew(subp, subt):
        if subt >= tslot[0]:
            idx = np.argmin(abs(tslot - subt))
            return (uslot[idx])
        else:
            return (h(p, subt))

    for i in range(n-1):
        tt = tslot[i]
        uu = uslot[i]
        k1 = delta * f(uu, hnew, p, tt)
        k2 = delta * f(uu + 0.5 * k1, hnew, p, tt+0.5 * delta)
        k3 = delta * f(uu+0.5 * k2, hnew, p, tt+0.5 * delta)
        k4 = delta * f(uu+k3, hnew, p, tt+delta)
        uslot[i+1] = uu + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return ((tslot, uslot))

