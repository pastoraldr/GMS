import math
import numpy as np
from scipy import interpolate
from diffeqpy import de


a = [0,3.68421053,  7.36842105, 11.05263158, 14.73684211, 18.42105263,
      22.10526316, 25.78947368, 29.47368421, 33.15789474, 36.84210526, 40.52631579,
      44.21052632, 47.89473684, 51.57894737, 55.26315789, 58.94736842, 62.63157895,
      66.31578947, 70]
b=  [0.42935087,  1.92088671,  0.41712295, -0.25007707, -0.18954231, -1.24214212,
     -1.07214407, -0.91995987,  0.51654649, -1.25133288,  0.29660553, -0.30552719,
     0.29201498,  1.10732322, -0.18839248, -0.99940753, -1.95084375,  0.91911038,
     -1.14697187,  0.86489447]


u0 =  [5.0, 2.0]
T = 70.0
tspan = (25.0, T)   # must be tuple
parameter =[10,10]+a+b

def history(p, t):
    return ([5.0, 2.0])


def mmodel(u, h, p, t):
    tmp1 = history(p, t - p[0])
    tmp2 = history(p, t - p[1])
    if float(t) < p[2] or float(t) > p[21]:
        v = 0.0
    else:
        v = interpolate.PchipInterpolator(p[2:21], p[22:41])(t)

    du1 = -u[0] + tmp2[1] + v
    du2 = -u[1] + tmp1[0]
    return ([du1, du2])

prob = de.DDEProblem(mmodel, u0, history, tspan, p= parameter)

sol = de.solve(prob)

print(sol.t)
print(sol.u)
