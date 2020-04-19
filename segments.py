import math
import numpy as np
from scipy import interpolate
from diffeqpy import de

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### some functions

def f1(x,a,r):
  v = x**r/(a**r + x**r)
  return (v)

##########################################################

def f2(x,a):
  v = x/(a + x)
  return (v)

##########################################################

def f3(x):
  v = x
  return (v)

##########################################################

def f4(x,a):
  v = x*x/(a*a + x*x)
  return (v)

##########################################################

def f5(x,a,r):
  v = a**r/(a**r + x**r)
  return (v)

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### function to determine meal function parameter


def DeterminMealFunctionParameters (t):
    THREE_HOURS = 180.0
    SEVEN_HOURS = 60.0 * 7.0
    TEN_HOURS = 60.0 * 10.0
    TWELVE_HOURS = 60.0 * 12.0
    FIFTHEEN_HOURS = 60.0 * 15.0
    EIGHTEEN_HOURS = 60.0 * 18.0
    TWEENTYTWO_HOURS = 60.0 * 22.0
    ONE_DAY = 60.0 * 24.0

    a = 0.0
    C = 0.0
    PeakTime = 10.0
    DigestTime = THREE_HOURS
    y = DigestTime + 1.0

    s = t
    while t > ONE_DAY:
        s = t % ONE_DAY
        t = t - ONE_DAY


    if s >= SEVEN_HOURS and s < (SEVEN_HOURS + DigestTime):
        a = 0.075
        C = 1.0
        PeakTime = 20.0
        y = s - SEVEN_HOURS
    elif s >= TEN_HOURS and s < (TEN_HOURS  + DigestTime):
        a = 0.025
        C = 1.0
        PeakTime = 10.0
        y = s - TEN_HOURS
    elif s >= TWELVE_HOURS and s < (TWELVE_HOURS  + DigestTime):
        a = 0.1
        C = 1.0
        PeakTime = 30.0
        y = s - TWELVE_HOURS
    elif s >= FIFTHEEN_HOURS and s < (FIFTHEEN_HOURS  + DigestTime):
        a = 0.025
        C = 1.0
        PeakTime = 10.0
        y = s - FIFTHEEN_HOURS
    elif s >= EIGHTEEN_HOURS and s < (EIGHTEEN_HOURS  + DigestTime):
        a = 0.11
        C = 1.0
        PeakTime = 35.0
        y = s - EIGHTEEN_HOURS
    elif s >= TWEENTYTWO_HOURS and s < (TWEENTYTWO_HOURS  + DigestTime):
        a = 0.01
        C = 1.0
        PeakTime = 15.0
        y = s - TWEENTYTWO_HOURS

    return ( [y, a, C, PeakTime, DigestTime] )


##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### Injection function

def InjectionFunction(t, a):
  if t % a < a/2:
    Iin = 0.5
  else:
    Iin = 0.1
  return(Iin)

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### Meal function

def MealFunction(t):
    para = DeterminMealFunctionParameters(t)
    y = para[0]
    a = para[1]
    C = para[2]
    PeakTime = para[3]
    DigestTime = para[4]

    if y > DigestTime or y < 0:
        Gin = 0
    else:
        Gin = max(0, a * C * y * math.exp(-y / PeakTime))

    return (Gin)


##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### Make history data from previous solutions

def SoltoInit (intime, insol,delay,time):
  sol_G = insol[0]
  sol_I = insol[1]
  solend_time = np.linspace(time-delay, time-1, delay)
  solend_G = 0 * solend_time
  solend_I = solend_G
  for k in range(delay):
    idx = np.argmin(abs(solend_time[k]-intime))
    solend_G[k] =  sol_G[idx]
    solend_I[k] =  sol_I[idx]
  return(list(solend_time) + list(solend_G) + list(solend_I))


##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### history function

def history(p,t):
  MaxDelay = max(p[0:3])
  if p[16]==0:
    if t-MaxDelay < 0:
      y =  [0.0, 0.0]
    elif t>24.0 & t<=36.0:
      y =  [85.0,40.0];
    elif t>12.0 & t<=24.0:
      y = [90.0, 75.0]
    elif t>=0 & t<=12:
      y = [100.0, 60.0]
    else:
      y = [1000.0, 1000.0]
  else:
    time = p[17:(17+MaxDelay-1)]
    Ghis = p[(17+MaxDelay):(17+2*MaxDelay-1)]
    Ihis = p[(17+2*MaxDelay):(17+3*MaxDelay-1)]
    pos = np.argmin(abs(np.array(time)-t))
    y = [Ghis[pos],Ihis[pos]]

  return (y)

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### model function

def mmodel(u, h, p, t):
    G = u[0]
    I = u[1]

    tau1 = p[0]
    tau2 = p[1]
    tau3 = p[2]

    VP_Gin = p[3]
    VP_di = p[4]

    ## parameters for f1
    VP_sig1 = p[5]
    VP_a1 = p[6]
    VP_r1 = p[7]

    ## parameters for f2
    VP_sig2 = p[8]
    VP_a2 = p[9]

    ## parameters for f3 and f4
    VP_sig4 = p[10]
    VP_U0 = p[11]
    VP_a4 = p[12]

    ## parameters for f5
    VP_sig5 = p[13]
    VP_a5 = p[14]
    VP_r5 = p[15]

    ##########
    G_tau1 = h(p, t - tau1)
    I_tau2 = h(p, t - tau2)
    I_tau3 = h(p, t - tau3)

    SwitchTimeOfInjection = 60;

    du1 = MealFunction(t) - VP_sig2 * G - VP_sig4 * G * (VP_U0 + I_tau3[1]) + VP_sig5 * (100 - I_tau2[1])
    du2 = InjectionFunction(t, SwitchTimeOfInjection) + 0.03 * VP_sig1 * f1(G_tau1[0], VP_a1, VP_r1) - VP_di * I

    return ([du1, du2])


##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### initiate  function

def initData(delay):
  idx = np.linspace(0,delay,delay+1)
  G = np.linspace(0,delay,delay+1)
  I = np.linspace(0,delay,delay+1)
  t = idx-delay
  for k in range(delay+1):
    G[k] = 100+6*math.sqrt(k)
    I[k] = 20+k/2
  n = len(idx)
  u = [G[n-1], I[n-1]]
  pu = np.concatenate([np.delete(t,n-1), np.delete(G,n-1), np.delete(I,n-1)])
  return(pu, u)


##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#### main
VP_tau1= 5
VP_tau2 = 36
VP_tau3 = 3

## parameters for insulin secretion triggered by glucose
VP_Gin = 0.5787
VP_di = 0.04

## parameters for f1
VP_sig1 = 75.0
VP_a1 = 203.0
VP_r1 = 4.0

## parameters for f2
VP_sig2 = 0.24
VP_a2 = 9.0

## parameters for f3 and f4
VP_sig4 = 0.009
VP_U0 = 0.4
VP_a4 = 50.6

## parameters for f5
VP_sig5 = 1.77
VP_a5 = 26.0
VP_r5 = 8.0

Delay = [VP_tau1,  VP_tau2,  VP_tau3]
MaxDelay = max(Delay)

DefaultStartTime = 0.0
DefaultVPEndTime = 1000.0
tspan = (DefaultStartTime, DefaultVPEndTime)

vpinit = [VP_tau1, VP_tau2, VP_tau3, VP_Gin, VP_di,
           VP_sig1, VP_a1, VP_r1,
           VP_sig2, VP_a2,
           VP_sig4, VP_U0, VP_a4,
           VP_sig5, VP_a5, VP_r5,1]

tmph, tmp0 =initData(MaxDelay)
vp0 = vpinit+list(tmph)
u0 = tmp0
prob0 = de.DDEProblem(mmodel, u0, history, tspan, p = vp0)
sol0 = de.solve(prob0)

##############################################
vp1 = vp0
u1 = u0
tspan1 = (DefaultStartTime, 500)
prob1 = de.DDEProblem(mmodel, u1, history, tspan1, p = vp1)
sol1= de.solve(prob1)
ut1 = np.transpose(sol1.u)

tmp = SoltoInit(sol1.t, ut1, MaxDelay, 500)
vp2 = vpinit+list(tmp)
vp2[5] = 1.2*vp2[5]
u2 = sol1.u[len(sol1.u)-1]
tspan2 = (500, 1000)
prob2 = de.DDEProblem(mmodel, u2, history, tspan1, p = vp2)
sol2 = de.solve(prob2)
ut2 = np.transpose(sol2.u)
