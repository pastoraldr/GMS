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
    return( np.array( [t+delta]+list(nu) ))



def rk_fixed(f, u, h, p, ts, k):
    ### fixed step size, roughly 1/k
    tslot = np.linspace(ts[0], ts[1], round((ts[1] - ts[0]) * k))
    n  = len(tslot)
    m = len(u)+1
    uslot = np.zeros( (n, m) )
    uslot[0][0] = ts[0]
    uslot[0][1:m] = u
    delta = tslot[1] - tslot[0]
    def hnew(subp, subt):
        if subt >= tslot[0]:
            idx = np.argmin(abs(tslot - subt))
            return (uslot[idx])
        else:
            return (h(p, subt))

    for i in range(n-1):
        tt = tslot[i]
        uu = uslot[i][1:m]
        uslot[i+1] = rk(f, uu, hnew, p, tt ,delta)
    return (uslot)




def rk_adaptive(f, u, h, p, ts, abserr):

    tsol = ts[0]
    tend = ts[1]
    m = len(u)+1
    p0 = 4
    p21 = 2 ** p0 - 1

    usol = np.array([tsol] + list(u))
    def hnew(subp, subt):
        if subt >= tslot[0]:
            idx = np.argmin(abs(tslot - subt))
            return (uslot[idx])
        else:
            return (h(p, subt))

    tt = usol[0]
    uu = usol[1:m]
    while tt < tend:
        delta = 0.1
        tmp1 = rk(f, uu, hnew, p, tt, delta)
        tmp2 = rk(f, uu, hnew, p, tt, delta / 2)
        if  sum(abs(tmp1[1:m]-tmp2[1:m])) / p21 < abserr :
            while sum(abs(tmp1[1:m]-tmp2[1:m])) / p21 < abserr :
                tmp2 = tmp1
                delta = 2 * delta
                tmp1 = rk(f, uu, hnew, p, tt, delta)

        else:
            while sum(abs(tmp1[1:m]-tmp2[1:m])) / p21 > abserr :
                tmp1 = tmp2
                delta = delta / 2
                tmp2 = rk(f, uu, hnew, p, tt, delta)


        tmp = rk(f, uu, hnew, p, tt, delta)
        tt = tmp[0]
        uu = tmp[1:m]
        usol =  np.vstack([usol, tmp])

    return (usol)
