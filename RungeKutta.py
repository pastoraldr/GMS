import math
import numpy as np
from scipy import interpolate

##### f dde model
##### u initial value
##### h history function
##### p parameter
##### t the current time point
##### delta the step

def rk(f, u, h, p, t ,delta):
  ### the return of f must be an array
  k1  = delta*f(u, h, p, t)
  k2 = delta* f(u+0.5*k1, h, p, t+0.5*delta)
  k3 = delta* f(u+0.5*k2, h, p, t+0.5*delta)
  k4 = delta* f(u+k3, h, p, t+delta)
  nu = u+(k1 + k2 + k2 + k3 + k3 + k4) / 6
  return(nu)





def rk_fixed(f, u, h, p, ts, k):
    ##### fixed step size
    ##### k  1/k is rougly the step size
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
        uslot[i+1] = rk(f, uu, hnew, p, tt ,delta)
    return ((tslot, uslot))


