#!/usr/bin/env python3
# jj_ap.py -- functions to calculate the action potential from a
# single Josephson junction neuron (JJN).  Used to generate the results
# shown in Crotty, Segall, and Schult, "Biologically realistic behaviors
# from a superconducting neuron model"
# requirements:  Python 3, numpy, scipy, matplotlib, pylab, TeX (for plotting)

# ----------------------------- COPYRIGHT INFO ------------------------|----- #
# Copyright (C) 2021 Patrick Crotty                                           #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
# ---------------------------------------------------------------------|----- #



# --------------------------------- MODULES ---------------------------|----- #
import base64
import copy
from collections import namedtuple
## named tuple class for JJN circuit parameters -- note lambda is called
## Lambda to avoid confusion with Python 'lambda' keyword
JJN_Param_Names = ['eta', 'Gamma', 'Lambda', 'Lambda_s', 'Lambda_p']
JJN_Param = namedtuple('JJN_Param', JJN_Param_Names)
## named tuple classes for figure
SL_Param_Names = ['ax', 'sl', 'axtb', 'tb']
SL_Param = namedtuple('SL_Param', SL_Param_Names)
B_Param_Names = ['ax', 'b']
B_Param = namedtuple('B_Param', B_Param_Names)
import io
import lzma
## if no display, set matplotlib backend appropriately
import matplotlib, os
if not os.environ.get('DISPLAY', ''):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
## uncomment below to set matplotlib to use TeX, sans-serif font
plt.rc('text', usetex=True)
#plt.rc('font', family='sans-serif')
import numpy as np
import random
import scipy.integrate as ode
import sys
import uuid
# ---------------------------------------------------------------------|----- #




# -------------------------------- CONSTANTS --------------------------|----- #
## default ODE integrator tolerances (absolute and relative) ##
DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 1e-12
## default JJN circuit parameters:
DEFAULT_PARAM = JJN_Param(1.0, 0.1, 0.01, 0.5, 0.5)
DEFAULT_I_BIAS_PARAM = (1.0,)
DEFAULT_I_IN_PARAM = (0,)
DEFAULT_T_START = 0.0
DEFAULT_T_END = 2000.0
## supported image file compression/encoding algorithms and formats
SUPPORTED_IMAGE_FILE_CMPENC = ('b64', 'xz')
SUPPORTED_IMAGE_FILE_FORMATS = ('eps', 'pdf', 'png', 'svg', 'jpg')
# ---------------------------------------------------------------------|----- #




# --------------------------------- CLASSES ---------------------------|----- #
class I_Constant:
    '''Current with constant value.
    fields:  Param_Names (class):  b: current level.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             p:  Param named tuple of parameter values.'''
    Param_Names = ['b']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = '$b$'
    
    def __init__(self, b=0.0):
        '''Initializes the current object.
        inputs:  b:  constant current level.  (Default:  0)
        output:  defines object field p with value of b.
        errors:  none.'''
        
        self.p = I_Constant.Param(b)

    
    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time (not used).
        ouptut:  returns the constant current level (b).
        errors:  none.'''

        return self.p.b



class I_Sin:
    '''Sinusoidally oscillating current.
    fields:  Param_Names (class):  T:  period; A:  amplitude;
               i0:  offset.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             p:  Param named tuple of parameter values.'''
    Param_Names = ['T', 'A', 'i0']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = '$i_0 + A \sin(2 \pi t / T)$'

    def __init__(self, T=50.0, A=0.2, i0=0.0):
        '''Initializes the current object.
        inputs:  T:  sine wave period.  (Default:  50.0)
                 A:  sine wave amplitude.  (Default:  0.2)
                 i0:  sine wave offset, i.e., current value at zero-
                   crossings of sine wave.  (Default:  0.0)
        output:  defines object field p with values of T, A, i0.
        errors:  none.'''
        
        self.p = I_Sin.Param(T, A, i0)


    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  i0 + A * sin(2 pi t / T).
        errors:  none.'''
        
        return (self.p.i0 + self.p.A * np.sin(2.0 * np.pi * t / self.p.T))



class I_Pulse:
    '''A single square pulse of current.
    fields:  Param_Names (class):  b:  constant current level during
               pulse; D:  pulse duration; tp:  pulse start time.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.              
             p:  Param named tuple of parameter values.'''    
    Param_Names = ['b', 'D', 'tp']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = '$b$ if $t_p \le t < t_p + D$; 0  otherwise'

    def __init__(self, b=0.0, D=0.0, tp=0.0):
        '''Initializes the current object.
        inputs:  b:  current level during pulse.  (Default:  0.0)
                 D:  pulse duration.  (Default:  0.0)
                 tp:  time at which pulse starts.  (Default:  0.0)
        output:  defines object field p with values of b, D, tp.
        errors:  none.'''
                
        self.p = I_Pulse.Param(b, D, tp)


    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  b if tp <= t < tp + D;
                   0 otherwise.
        errors:  none.'''
        
        if (t < self.p.tp) or (t >= self.p.tp + self.p.D):
            return 0.0
        else:
            return self.p.b



class I_2EqPulses:
    '''Two equal pulses of current.
    fields:  Param_Names (class):  b:  constant current level during 
               each pulse; D:  pulse duration; tp1, tp2:  pulse start 
               times.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             tp:  array of pulse start times.
             _p:  Param named tuple of parameter values. (Internal)
    properties:  p:  _p (External)'''
    Param_Names = ['b', 'D', 'tp1', 'tp2']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = ('$b$ if $t_{pn} \le t < t_{pn} + D$ for $n = 1,2$; overlaps add;'
              ' 0 otherwise')

    def __init__(self, b=0.0, D=0.0, tp1=0.0, tp2=0.0):
        '''Initializes the current object.
        inputs:  b:  current level during pulse.  (Default:  0.0)
                 D:  pulse duration.  (Default:  0.0)
                 tp1:  time at which first pulse starts.  (Default:  0.0)
                 tp2:  time at which second pulse starts.  (Default:  0.0)
        output:  defines object field _p with values of b, D, tp1, tp2,
                 and array of pulse start times tp.
        errors:  none.'''

        self._p = I_2EqPulses.Param(b, D, tp1, tp2)
        self.tp = [tp1, tp2]

        
    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  0.0 if not during a pulse,
                   and b if so.  If t is in the overlap of multiple pulses,
                   returns the corresponding multiple of b values.
        errors:  none.'''

        retval = 0.0
        for tpn in self.tp:
            if (t >= tpn) and (t < tpn + self.p.D):
                retval = retval + self.p.b
        return retval


    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, newval):
        self._p = newval
        self.tp = [self.p.tp1, self.p.tp2]



class I_4EqPulses:
    '''Four equal pulses of current.
    fields:  Param_Names (class):  b:  constant current level during 
               each pulse; D:  pulse duration; tp1..tp4:  pulse start 
               times.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             tp:  array of pulse start times.
             _p:  Param named tuple of parameter values. (Internal)
    properties:  p:  _p (External)'''
    Param_Names = ['b', 'D', 'tp1', 'tp2', 'tp3', 'tp4']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = ('$b$ if $t_{pn} \le t < t_{pn} + D$ for $n = 1,2,3,4$; overlaps'
              ' add; 0 otherwise')

    def __init__(self, b=0.0, D=0.0, tp1=0.0, tp2=0.0, tp3=0.0, tp4=0.0):
        '''Initializes the current object.
        inputs:  b:  current level during pulse.  (Default:  0.0)
                 D:  pulse duration.  (Default:  0.0)
                 tp1:  time at which first pulse starts.  (Default:  0.0)
                 tp2:  time at which second pulse starts.  (Default:  0.0)
                 tp3:  time at which third pulse starts.  (Default:  0.0)
                 tp4:  time at which fourth pulse starts.  (Default:  0.0)
        output:  defines object field _p with values of b, D, tp1..tp4,
                 and array of pulse start times tp.
        errors:  none.'''

        self._p = I_4EqPulses.Param(b, D, tp1, tp2, tp3, tp4)
        self.tp = [tp1, tp2, tp3, tp4]

        
    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  0.0 if not during a pulse,
                   and b if so.  If t is in the overlap of multiple pulses,
                   returns the corresponding multiple of b values.
        errors:  none.'''

        retval = 0.0
        for tpn in self.tp:
            if (t >= tpn) and (t < tpn + self.p.D):
                retval = retval + self.p.b
        return retval


    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, newval):
        self._p = newval
        self.tp = [self.p.tp1, self.p.tp2, self.p.tp3, self.p.tp4]


        
class I_2AltPulses:
    '''Two equal-magnitude pulses of current, with the second having 
    opposite sign to the first.
    fields:  Param_Names (class):  b:  constant current level magnitude 
               during each pulse; D:  pulse duration; tp1, tp2:  pulse start 
               times.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             tp:  array of pulse start times.
             _p:  Param named tuple of parameter values. (Internal)
    properties:  p:  _p (External)'''
    Param_Names = ['b', 'D', 'tp1', 'tp2']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = ('$(-1)^{(n+1)} \, b$ if $t_{pn} \le t < t_{pn} + D$ for'
              ' $n = 1,2$; overlaps add; 0 otherwise')

    def __init__(self, b=0.0, D=0.0, tp1=0.0, tp2=0.0):
        '''Initializes the current object.
        inputs:  b:  current level magnitude during pulse.  (Default:  0.0)
                 D:  pulse duration.  (Default:  0.0)
                 tp1:  time at which first pulse starts.  (Default:  0.0)
                 tp2:  time at which second pulse starts.  (Default:  0.0)
        output:  defines object field _p with values of b, D, tp1, tp2,
                 and array of pulse start times tp.
        errors:  none.'''

        self._p = I_2AltPulses.Param(b, D, tp1, tp2)
        self.tp = [tp1, tp2]

        
    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  0.0 if not during a pulse,
                   -b during the first pulse, +b during the second.  
                   If t is in the overlap of the pulses, returns 0.
        errors:  none.'''

        retval = 0.0
        if (t >= self.p.tp1) and (t < self.p.tp1 + self.p.D):
            retval = retval - self.p.b
        if (t >= self.p.tp2) and (t < self.p.tp2 + self.p.D):
            retval = retval + self.p.b
        return retval


    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, newval):
        self._p = newval
        self.tp = [self.p.tp1, self.p.tp2]



class I_3AltPulses:
    '''Three equal-magnitude pulses of current, with the second having 
    opposite sign to the first and third.
    fields:  Param_Names (class):  b:  constant current level magnitude 
               during each pulse; D:  pulse duration; tp1..tp3:  pulse start 
               times.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             tp:  array of pulse start times.
             _p:  Param named tuple of parameter values. (Internal)
    properties:  p:  _p (External)'''
    Param_Names = ['b', 'D', 'tp1', 'tp2', 'tp3']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = ('$(-1)^{(n+1)} \, b$ if $t_{pn} \le t < t_{pn} + D$ for '
              '$n = 1,2,3$; overlaps add; 0 otherwise')

    def __init__(self, b=0.0, D=0.0, tp1=0.0, tp2=0.0, tp3=0.0):
        '''Initializes the current object.
        inputs:  b:  current level magnitude during pulse.  (Default:  0.0)
                 D:  pulse duration.  (Default:  0.0)
                 tp1:  time at which first pulse starts.  (Default:  0.0)
                 tp2:  time at which second pulse starts.  (Default:  0.0)
                 tp3:  time at which third pulse starts.  (Default:  0.0)
        output:  defines object field _p with values of b, D, tp1..tp3,
                 and array of pulse start times tp.
        errors:  none.'''

        self._p = I_3AltPulses.Param(b, D, tp1, tp2, tp3)
        self.tp = [tp1, tp2, tp3]

        
    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  0.0 if not during a pulse,
                   b during the first or third pulse, -b during the second.  
                   If t is in the overlap of multiple pulses,
                   returns the corresponding sum of b values.
        errors:  none.'''

        retval = 0.0
        if (t >= self.p.tp1) and (t < self.p.tp1 + self.p.D):
            retval = retval + self.p.b
        if (t >= self.p.tp2) and (t < self.p.tp2 + self.p.D):
            retval = retval - self.p.b
        if (t >= self.p.tp3) and (t < self.p.tp3 + self.p.D):
            retval = retval + self.p.b
        return retval


    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, newval):
        self._p = newval
        self.tp = [self.p.tp1, self.p.tp2, self.p.tp3]



class I_Step:
    '''A linearly-rising or falling step of current.
    fields:  Param_Names (class):  ts:  step start time; b:  step
               initial current level; m:  step slope; D;  step
               duration.
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             p:  Param named tuple of parameter values.'''    
    Param_Names = ['ts', 'b', 'm', 'D']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = '$b + m(t - t_s)$ for $t_s \le t < t_s + D$; 0 otherwise'

    def __init__(self, ts=0.0, b=0.0, m=0.0, D=0.0):
        '''Initializes the current object.
        inputs:  ts:  time at which step starts.  (Default:  0.0)  
                 b:  current level just after step starts.  (Default:  0.0)
                 m:  linear slope of current during step.  (Default:  0.0)
                 D:  step duration.  (Default:  0.0)
        output:  defines object field p with values of ts, b, m, D.
        errors:  none.'''

        self.p = I_Step.Param(ts, b, m, D)


    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  b + m * (t - ts) if 
                   ts <= t < ts + D; 0 otherwise.
        errors:  none.'''

        if (t < self.p.ts) or (t >= self.p.ts + self.p.D):
            return 0.0
        else:
            return self.p.b + self.p.m * (t - self.p.ts)



class I_StepPulse:
    '''A linearly-rising or falling step of current followed by a square
    pulse.
    fields:  Param_Names (class):  ts:  step start time; bs:  step
               initial current level; ms:  step slope; Ds;  step
               duration; tp:  pulse start time; bp:  pulse current level; 
               Dp:  pulse duration.             
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             p:  Param named tuple of parameter values.'''    
    Param_Names = ['ts', 'bs', 'ms', 'Ds', 'tp', 'bp', 'Dp']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = ('$b_s + m_s(t - t_s)$ for $t_s \le t < t_s + D_s$;'
              ' $b_p$ for $t_p \le t < t_p + D_p$; overlaps add; 0 otherwise')

    def __init__(self, ts=0.0, bs=0.0, ms=0.0, Ds=0.0, tp=0.0,
                 bp=0.0, Dp=0.0):
        '''Initializes the current object.
        inputs:  ts:  time at which step starts.  (Default:  0.0)  
                 bs:  current level just after step starts.  (Default:  0.0)
                 ms:  linear slope of current during step.  (Default:  0.0)
                 Ds:  step duration.  (Default:  0.0)
                 tp:  pulse start time.  (Default 0.0)
                 bp:  pulse current level.  (Default:  0.0)
                 Dp:  pulse duration.  (Default:  0.0)
        output:  defines object field p with values of ts, bs, ms, Ds,
                 tp, bp, Dp.
        errors:  none.'''

        self.p = I_StepPulse.Param(ts, bs, ms, Ds, tp, bp, Dp)


    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  bs + ms * (t - ts) if 
                   ts <= t < ts + Ds; 0 bp if tp <= t < tp + Dp;
                   the sum of these if they overlap; 0 otherwise.
        errors:  none.'''

        retval = 0.0
        if (t >= self.p.ts) and (t < self.p.ts + self.p.Ds):
            retval = retval + self.p.bs + self.p.ms * (t - self.p.ts)
        if (t >= self.p.tp) and (t < self.p.tp + self.p.Dp):
            retval = retval + self.p.bp
        return retval



class I_StepStep:
    '''Two successive linearly-rising or falling steps of current.
    fields:  Param_Names (class):  ts1:  step 1 start time; bs1 :  step
               1 initial current level; ms1:  step 1 slope; Ds1;  step 1
               duration; ts2:  step 2 start time; bs2 :  step
               2 initial current level; ms2:  step 2 slope; Ds2;  step 2
               duration.              
             Param (class):  named tuple type of Param_Names.
             Eq_Str (class):  mathematical formula for current.
             p:  Param named tuple of parameter values.'''    
    Param_Names = ['ts1', 'bs1', 'ms1', 'Ds1', 'ts2', 'bs2', 'ms2', 'Ds2']
    Param = namedtuple('Param', Param_Names)
    Eq_Str = ('$b_{sn} + m_{sn}(t - t_{sn})$ for '
              '$t_{sn} \le t < t_{sn} + D_{sn}$ for $n = 1,2$; overlaps '
              'add; 0 otherwise')

    def __init__(self, ts1=0.0, bs1=0.0, ms1=0.0, Ds1=0.0, ts2=0.0,
                 bs2=0.0, ms2=0.0, Ds2=0.0):
        '''Initializes the current object.
        inputs:  ts1:  time at which step 1 starts.  (Default:  0.0)  
                 bs1:  current level just after step 1 starts.  (Default:  0.0)
                 ms1:  linear slope of current during step 1.  (Default:  0.0)
                 Ds1:  step 1 duration.  (Default:  0.0)
                 ts2:  time at which step 2 starts.  (Default:  0.0)  
                 bs2:  current level just after step 2 starts.  (Default:  0.0)
                 ms2:  linear slope of current during step 2.  (Default:  0.0)
                 Ds2:  step 2 duration.  (Default:  0.0)
        output:  defines object field p with values of ts1, bs1, ms1, Ds1,
                 ts2, bs2, ms2, Ds2.
        errors:  none.'''

        self.p = I_StepStep.Param(ts1, bs1, ms1, Ds1, ts2, bs2, ms2, Ds2)


    def __call__(self, t):
        '''Calls the object as a function.
        inputs:  t:  time.
        ouptut:  returns the current at t:  bs1 + ms1 * (t - ts1) if 
                   ts1 <= t < ts1 + Ds1; similar for the second step;
                   the sum of these if they overlap; 0 otherwise.
        errors:  none.'''

        retval = 0.0
        if (t >= self.p.ts1) and (t < self.p.ts1 + self.p.Ds1):
            retval = retval + self.p.bs1 + self.p.ms1 * (t - self.p.ts1)
        if (t >= self.p.ts2) and (t < self.p.ts2 + self.p.Ds2):
            retval = retval + self.p.bs2 + self.p.ms2 * (t - self.p.ts2)
        return retval



class JJN:
    '''A Josephson junction neuron.
    fields:  p:  a JJN_Param named tuple (eta, Gamma, lambda, Lambda_s,
               Lambda_p).
             i_bias:  current object for the bias current.
             i_in:  current object for the input current.'''

    def __init__(self, param=DEFAULT_PARAM, i_bias=I_Constant(b=2.2),
                 i_in=I_Step()):
        '''Initializes the JJN object.
        inputs:  param:  A JJN_Param named tuple of parameter values.
                   (Default:  DEFAULT_PARAM)
                 i_bias:  bias current object.  (Default:  I_Constant
                   with b = 2.2)
                 i_in:  input current object.  (Default:  I_Step
                   with defaults)
        output:  sets p, i_bias, and i_in fields respectively.
        errors:  none.'''
        
        self.p = param
        self.i_bias = i_bias
        self.i_in = i_in
# ---------------------------------------------------------------------|----- #




# ----------------------------------- DATA ----------------------------|----- #
## izh_param:  dict containing parameter values found for specific
##   Izhikevich behaviors.  Each entry is a tuple (JJN, t_final) where
##   JJN is a JJN object (see above) and t_final is the total
##   integration time.  The keys are codes for the behaviors:
##   TS:  tonic spiking
##   PS:  phasic spiking
##   TB:  tonic bursting
##   PB:  phasic bursting
##   MM:  mixed mode
##   SFA:  spike frequency adaptation
##   C1:  class 1 excitable
##   C2:  class 2 excitable
##   SL:  spike latency
##   STO:  subthreshold oscillations
##   RS:  rebound spike
##   RB:  rebound burst
##   DAP:  depolarizing afterpotential
##   IIS:  inhibition-induced spiking
##   IIB:  inhibition-induced bursting
##   BI:  bistability
##   RE:  resonator
##   IN:  integrator
##   TV:  threshold variability
izh_param = {}
izh_param['TS'] = (JJN(param=JJN_Param(1.2, 1.53, 0.16, 0.35, 0.55),
                       i_bias=I_Constant(2.19),
                       i_in=I_Pulse(0.13, 1000, 200)), 2000)
izh_param['PS'] = (JJN(param=JJN_Param(1.71, 1.55, 0.13, 0.49, 0.48),
                       i_bias=I_Constant(2.085),
                       i_in=I_Pulse(0.13, 600, 200)), 1000)
izh_param['TB'] = (JJN(param=JJN_Param(0.95, 1.81, 0.11, 0.44, 0.46),
                       i_bias=I_Sin(200, 0.2, 1.73),
                       i_in=I_Pulse(0.57, 1000, 200)), 2000)
izh_param['PB'] = (JJN(param=JJN_Param(1.58, 1.64, 0.09, 0.61, 0.1),
                       i_bias=I_Constant(0.95),
                       i_in=I_Pulse(2.9, 300, 200)), 600)
izh_param['MM'] = (JJN(param=JJN_Param(0.9410, 1.68329, 0.03, 0.49, 0.48),
                       i_bias=I_Constant(1.928),
                       i_in=I_Pulse(1.75949, 1000, 200)), 2000)
izh_param['SFA'] = (JJN(param=JJN_Param(1.79, 2.67, 0.001, 0.52, 0.48),
                        i_bias=I_Constant(1.9),
                        i_in=I_Pulse(0.13, 1000, 200)), 2000)
izh_param['C1'] = (JJN(param=JJN_Param(0.96, 1.55, 0.26, 0.49, 0.48),
                       i_bias=I_Constant(1.926),
                       i_in=I_Step(200, 0.13, 5.33e-4, 626.67)), 2000)
izh_param['C2'] = (JJN(param=JJN_Param(0.96, 1.04, 0.26, 0.46, 0.46),
                       i_bias=I_Constant(1.926),
                       i_in=I_Step(200, 0.13, 5.33e-4, 626.67)), 2000)
izh_param['SL'] = (JJN(param=JJN_Param(1.71, 1.55, 0.13, 0.49, 0.48),
                       i_bias=I_Constant(2.1433799999999987),
                       i_in=I_Pulse(0.03, 20, 460)), 2000)
izh_param['STO'] = (JJN(param=JJN_Param(1.71, 0.6494499, 0.129869,0.49, 0.48),
                        i_bias=I_Sin(8.3799, 0.00719, 1.909552),
                        i_in=I_Pulse(1.12739, 4.6479, 851.82)), 2000)
izh_param['RS'] = (JJN(param=JJN_Param(1.7082899999, 1.55, 0.12963079,
                                       0.51, 0.48),
                       i_bias=I_Constant(2.1433799),
                       i_in=I_Pulse(-2.498040, 10.4723799, 446.199)), 600)
izh_param['RB'] = (JJN(param=JJN_Param(0.95, 0.758389, 0.11, 0.44, 0.46),
                       i_bias=I_Sin(200, 0.2, 1.678099),
                       i_in=I_Pulse(-2.498040, 10.4723799, 446.199)), 600)
izh_param['DAP'] = (JJN(param=JJN_Param(1.2619, 1.593, 0.182649,
                                        0.54, 0.46),
                        i_bias=I_Constant(2.01692),
                        i_in=I_Pulse(0.74818, 5.3039, 46.0)), 200)
izh_param['IIS'] = (JJN(param=JJN_Param(1.2, 1.53, 0.16, 0.35, 0.55),
                        i_bias=I_Constant(2.19),
                        i_in=I_Pulse(-0.13, 1000, 200)), 2000)
izh_param['IIB'] = (JJN(param=JJN_Param(0.95, 1.81, 0.11, 0.44, 0.46),
                        i_bias=I_Sin(200, 0.2, 1.73),
                        i_in=I_Pulse(-0.57, 1000, 200)), 2000)
izh_param['BI'] =  (JJN(param=JJN_Param(1.0, 0.93, 0.064, 0.5, 0.46),
                            i_bias=I_Constant(1.906),
                            i_in=I_2EqPulses(0.55, 9, 100, 369.5)),
                        600)
izh_param['RE'] = (JJN(param=JJN_Param(1.71, 0.649, 0.13, 0.49, 0.48),
                       i_bias=I_Sin(2.0 * np.pi * 8.38, 0.012, 1.91),
                       i_in=I_4EqPulses(0.60, 4, 35, 45, 100, 117)), 300)
izh_param['IN'] = (JJN(param=JJN_Param(1.115, 2.1139, 0.1, 0.5, 0.5),
			i_bias=I_Constant(2.052618103027343),
			i_in=I_4EqPulses(0.669528, 1.6599, 400, 410, 600,
                                         650)), 1000)
izh_param['TV'] = (JJN(param=JJN_Param(1.1, 1.5, 0.1, 0.5, 0.5),
                       i_bias=I_Constant(1.96),
                       i_in=I_3AltPulses(0.3573, 10, 200, 495, 513.3)), 700)
## izh_param_list:  list of the keys in izh_param
izh_param_list = list(izh_param.keys())
# ---------------------------------------------------------------------|----- #




# -------------------------- CALCULATION FUNCTIONS --------------------|----- #
def get_range(pval):
    '''Calculates a new parameter slider range based on a given value.
    inputs:  pval:  a value.
    output:  returns (pval/10, 3*pval) if pval>0; (3*pval, pval/10)
             if pval<0; (-1,1) if pval=0.
    errors:  none.'''

    if pval == 0.0:
        return (-1.0, 1.0)
    elif pval < 0.0:
        return (pval * 3.0, pval / 10.0)
    else:
        return (pval / 10.0, pval * 3.0)



def int_jj1(jjn, t_end=DEFAULT_T_END, num_t=20001, atol=DEFAULT_ATOL,
            rtol=DEFAULT_RTOL):
    '''Integrates the equations of motion for a JJN.
    inputs:  jjn:  a JJN object.
             [t_end]:  total time span to integrate over.  (Default:
               DEFAULT_T_END)
             [num_t]:  number of evenly-spaced time points at which
               to calculate.  (Default:  5001)
             [atol]:  absolute tolerance for ODE integrator
               (Default:  DEFAULT_ATOL)
             [rtol]:  relative tolerance for ODE integrator
               (Default:  DEFAULT_RTOL)
    output:  a tuple (t, s_f, i_bias_arr, i_in_arr), where these are
             respectively arrays of time values, state vector values and flux,
             bias current values, and input current values.  Each row of
             s_f is [phi_p, phi_p', phi_c, phi_c', f], where
             f = lambda * (phi_p + phi_c).
    errors:  none.'''

    s_0 = np.zeros(4)
    t = np.linspace(0, t_end, num_t)
    s = ode.odeint(jj1, s_0, t, args=(jjn,), rtol=rtol, atol=atol)
    f = np.zeros(len(t))
    for j in range(len(t)):
        f[j] = jjn.p.Lambda * (s[j, 0] + s[j, 2])
    s_f = np.concatenate((s, f.reshape((1, len(f))).T), axis=1)
    i_in_arr = np.array([jjn.i_in(tel) for tel in t])
    i_bias_arr = np.array([jjn.i_bias(tel) for tel in t])
    return (t, s_f, i_bias_arr, i_in_arr)


    
def jj1(s, t, jjn):
    '''Derivatives function for a single, isolated Josephson junction 
    neuron.  All units arbitrary.
    inputs:  s:  state vector:  [phi_p, phi_p', phi_c, phi_c'] (list or array).
             t:  time at which to calculate derivatives.
             jjn:  a JJN object.
    output:  ds/dt at t (NumPy array).
    errors:  none.''' 

    p = jjn.p
    ds0dt = s[1]
    ds1dt = (-p.Gamma * s[1] - np.sin(s[0]) - p.Lambda * (s[2] + s[0]) +
             p.Lambda_s * jjn.i_in(t) + (1 - p.Lambda_p) * jjn.i_bias(t))
    ds2dt = s[3]
    ds3dt = (-p.Gamma * s[3] - np.sin(s[2]) + (1 / p.eta) *
             (-p.Lambda * (s[2] + s[0]) + p.Lambda_s * jjn.i_in(t) -
              p.Lambda_p * jjn.i_bias(t)))
    return np.array([ds0dt, ds1dt, ds2dt, ds3dt])



def mess_around(jjn=JJN(), index=1):
    '''Graphical tool to explore JJN parameter space.
    inputs:  jjn:  a JJN object.
             [index]:  index of the state vector to plot: 0 through 4
               correspond respectively to phi_p, phi_p', phi_c, phi_c',
               flux.  (Default:  1, phi_p')
    output:  creates a window that plots the JJ AP and allows parameter
             values to be modified.  The window contains the following
             elements:
             - a combined plot of the action potential (blue), input 
             current (red), and bias current (green).
             - entry fields for the start (t_start) and end(t_end) times
             of the integration, and the absolute (atol) and relative
             (rtol) tolerances for the ODE integrator.
             - sliders (blue) to modify the intrinsic JJN parameters.
             - slider(s) (green) to modify the bias current parameter(s).
             - slider(s) (red) to modify the input current parameter(s).
             - text entry fields for each of the parameters.
             - buttons to reset the values and currents to different
             Izhikevich behaviors.  These are labeled with the keys in the
             izh_param dict.
             - control buttons:  
               'phi_p', 'phi_p dot', 'phi_c', 'phi_c dot', and '(capital) Phi'
                 plot the corresponding quantity, with capital Phi flux.
               'W' saves the data in the plot to a text file in the same 
                 directory called 'jj_ap.out'.  Columns are 
                 (t, ap, bias, input).
               '!' readjusts the slider ranges to put the current values back
                 at the reference lines.
               '?' prints all current parameter values to the terminal.
               'Help' brings up a window explaning the controls.
    errors:  none.'''

    # subfunctions
    # Besides mess_around's arguments jjn and index, these subfunctions involve
    # the following mess_around-local variables, which are assigned in the
    # main function code and/or set via controls on the main window:
    #   atol, rtol:  absolute and relative tolerances for ODE integrator
    #   t_start, t_end:  time range of plot
    #   t_start_box, t_end_box:  handles for t_start and t_end input boxes
    #   ax, ax2, ax3:  axis handles
    #   p, p2, p3:  subplot handles
    #   sl, slb, sli:  slider handle dicts

    def change_index(val, new_index):
        '''Changes the state variable plotted.
        inputs:  val:  (dummy variable needed for binding)
                 new_index:  the new index value:  0: phi_p, 1:  phi_p',
                   2:  phi_c, 3:  phi_c', 4:  flux.
        output:  updates the index variable and replots.
        errors:  none.'''
        
        nonlocal index

        index = new_index
        update_plot(None)


    def display_help_window(vald):
        '''Displays a help window with information about the plotting controls.
        inputs:  vald:  (dummy variable needed for binding)
        output:  brings up a help window for the plotting controls.
        errors:  none.'''
        
        helpwin = plt.figure()
        helpwin.text(0.05, 0.95, 'Plotting Controls', fontsize=16)
        helpwin.text(0.05, 0.9, '$t_{start}$, $t_{end}$: plot start and end'
                     ' time', fontsize=12)
        helpwin.text(0.05, 0.85, '$atol$, $rtol$:  absolute and relative'
                     ' integration tolerances', fontsize=12)
        helpwin.text(0.05, 0.8, '$\eta$, $\Gamma$, $\Lambda$, $\Lambda_s$,'
                     ' $\Lambda_p$ (blue):  Josephson junction neuron circuit '
                     'parameters (note $\Lambda \equiv \lambda$)', fontsize=12)
        helpwin.text(0.05, 0.75, ', '.join(['$' + el + '$'
                                            for el in jjn.i_bias.Param_Names])
                     + ' (green):  bias current parameters:', fontsize=12)
        helpwin.text(0.1, 0.7, '$i_{bias}(t)$ = ' + jjn.i_bias.Eq_Str,
                     fontsize=12)
        helpwin.text(0.05, 0.65, ', '.join(['$' + el + '$' for el
                                            in jjn.i_in.Param_Names])
                     + ' (red):  input current parameters:', fontsize=12)
        helpwin.text(0.1, 0.6, '$i_{in}(t)$ = ' + jjn.i_in.Eq_Str, fontsize=12)
        helpwin.text(0.05, 0.55, 'TS, PS, ..., TV:  Izhikevich behaviors',
                     fontsize=12)
        helpwin.text(0.05, 0.5, '$\phi_p$, $\dot{\phi}_p$, $\phi_c$, '
                     '$\dot{\phi}_c$:  junction phase or derivative to plot',
                     fontsize=12)
        helpwin.text(0.05, 0.45, '$\Phi$:  plot magnetic flux '
                     '($\Phi = \Lambda (\phi_p + \phi_c)$)', fontsize=12)
        helpwin.text(0.05, 0.4, 'W:  write data on plot to jj\_ap.out',
                     fontsize=12)
        helpwin.text(0.05, 0.35, '!:  recenter sliders', fontsize=12)
        helpwin.text(0.05, 0.3, '?:  print current parameter values to '
                     'terminal', fontsize=12)
        helpwin.text(0.05, 0.25, 'Help:  this window', fontsize=12)
        helpwin.text(0.05, 0.05, 'See Crotty, Segall, Schult manuscript and '
                     'jj\_ap.py code for more details', fontsize=12)
        helpwin.show()

        
    def get_j_start(t):
        '''Finds the index in a time array corresponding to t_start.
        inputs:  t:  a time array (increasing order).
        output:  returns the lowest index j such that t[j] >= t_start,
                 or 0 if t_start is 0.
        errors:  returns -1 if t_start > t_end.'''
        
        if t_start == 0.0:
            return 0
        elif t_start > t_end:
            return -1
        else:
            j = 0
            while t[j] < t_start:
                j = j + 1
            return j


    def plot_data(t, s, i_bias_ret, i_in_ret):
        '''Plots/replots AP and current data.
        inputs:  t:  a time array, corresponding to the other arrays.
                 s:  an array of state vectors.
                 i_bias_ret:  array of bias current values.
                 i_in_ret:  array of input current values.
        output:  plots/replots the data, using the functions below to
                 determine axis ranges, etc.
        errors:  none.'''

        j_start = get_j_start(t)
        (yl, yu) = y_plot_range(s[j_start:, index], 'ap')
        (y2l, y2u) = y_plot_range(i_in_ret[j_start:], 'i_in')
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(yl, yu)
        if index == 0:
            ylabel = r"$\phi_p$ \, [a. u.]"
        elif index == 1:
            ylabel = r"$\dot{\phi}_p$ \, [a. u.]"
        elif index == 2:
            ylabel = r"$\phi_c$ \, [a. u.]"
        elif index == 3:
            ylabel = r"$\dot{\phi}_c$ \, [a. u.]"
        elif index == 4:
            ylabel = r"$\Phi$ \, [a. u.]"
        ax.set_ylabel(ylabel, color='blue')
        p.set_xdata(t[j_start:])
        p.set_ydata(s[j_start:, index])
        ax2.set_xlim(t_start, t_end)
        ax2.set_ylim(y2l, y2u)
        p2.set_xdata(t[j_start:])
        p2.set_ydata(i_in_ret[j_start:])
        ax3.set_xlim(t_start, t_end)
        ax3.set_ylim(y2l, y2u)
        p3.set_xdata(t[j_start:])
        p3.set_ydata(i_bias_ret[j_start:])
        plt.draw()


    def print_param_vals(val):
        '''Prints the current JJN and current parameters to the terminal.
        Invoked by the '?' button on the main window.
        inputs:  val:  (dummy variable needed for binding)
        output:  prints parameter values of the JJN, bias, and input
                 currents to the terminal.
        errors:  none.'''
        
        print('---current parameter values---')
        print('----JJN----')
        for pel in JJN_Param_Names:
            print(pel, sl.sl[pel].val)
        print('----i_bias----')
        for pel in jjn.i_bias.Param_Names:
            print(pel, slb.sl[pel].val)
        print('----i_in----')
        for pel in jjn.i_in.Param_Names:
            print(pel, sli.sl[pel].val)


    def reset_params(val, newconfig):
        '''Resets the JJN and current parameter values, and replots
        the AP and currents.  Invoked by the buttons for different
        Izhikevich behaviors.
        inputs:  val:  (dummy variable needed for binding)
                 newconfig:  new parameter values, in the format of
                   an entry in the izh_param global dict.
        output:  recalculates and replots data, reconfigures sliders.
        errors:  none.'''
        
        nonlocal jjn
        nonlocal sl
        nonlocal slb
        nonlocal sli
        nonlocal t_end

        for nel in JJN_Param_Names:
            sl.ax[nel].set_visible(False)
            sl.sl[nel].active = False
            sl.axtb[nel].set_visible(False)
            sl.tb[nel].active = False
        for nel in jjn.i_bias.Param_Names:
            slb.ax[nel].set_visible(False)
            slb.sl[nel].active = False
            slb.axtb[nel].set_visible(False)
            slb.tb[nel].active = False
        for nel in jjn.i_in.Param_Names:
            sli.ax[nel].set_visible(False)
            sli.sl[nel].active = False
            sli.axtb[nel].set_visible(False)
            sli.tb[nel].active = False
        sl = SL_Param({}, {}, {}, {})
        slb = SL_Param({}, {}, {}, {})
        sli = SL_Param({}, {}, {}, {})
        jjn = copy.deepcopy(newconfig[0])
        t_end = newconfig[1]
        (t, s, i_bias_ret, i_in_ret) = int_jj1(jjn, t_end=t_end, atol=atol,
                                               rtol=rtol)
        for nel in JJN_Param_Names:
            nelind = JJN_Param_Names.index(nel)
            newval = jjn.p[nelind]
            sl.ax[nel] = plt.axes([0.05 + nelind * 0.2, 0.2, 0.05, 0.025],
                                  label=uuid.uuid4()) 
            sl.sl[nel] = plt.Slider(sl.ax[nel], '$\\' + nel + '$',
                                    get_range(newval)[0], get_range(newval)[1],
                                    valinit=newval, facecolor='blue')
            sl.ax[nel].set_visible(True)
            sl.sl[nel].on_changed(lambda vald, eld=nel:
                                  update_values(vald, 's', 'jjnp', eld))  
            sl.axtb[nel] = plt.axes([0.05 + nelind * 0.2 + 0.05, 0.2, 0.05,
                                     0.025], label=uuid.uuid4())
            sl.tb[nel] = plt.matplotlib.widgets.TextBox(sl.axtb[nel], '',
                                                        initial = '%g' %
                                                        newval)
            sl.axtb[nel].set_visible(True)
            sl.tb[nel].on_submit(lambda vald, eld=nel:
                                 update_values(vald, 'tb', 'jjnp', eld))
        for nel in jjn.i_bias.Param_Names:
            nelind = jjn.i_bias.Param_Names.index(nel)
            newval = jjn.i_bias.p[nelind]
            slb.ax[nel] = plt.axes([0.05 + nelind * 0.2, 0.15, 0.05, 0.025],
                                   label=uuid.uuid4())
            slb.sl[nel] = plt.Slider(slb.ax[nel], '$' + nel + '$',
                                     get_range(newval)[0],
                                     get_range(newval)[1],
                                     valinit=newval, facecolor='green')
            slb.ax[nel].set_visible(True)
            slb.sl[nel].on_changed(lambda vald, eld=nel:
                                   update_values(vald, 's', 'ibias', eld))  
            slb.axtb[nel] = plt.axes([0.05 + nelind * 0.2 + 0.05, 0.15, 0.05,
                                      0.025], label=uuid.uuid4())
            slb.tb[nel] = plt.matplotlib.widgets.TextBox(slb.axtb[nel], '',
                                                         initial = '%g' %
                                                         newval)
            slb.axtb[nel].set_visible(True)
            slb.tb[nel].on_submit(lambda vald, eld=nel:
                                  update_values(vald, 'tb', 'ibias', eld))
        for nel in jjn.i_in.Param_Names:
            nelind = jjn.i_in.Param_Names.index(nel)
            newval = jjn.i_in.p[nelind]
            sli.ax[nel] = plt.axes([0.05 + nelind * 0.125, 0.1, 0.05, 0.025],
                                   label=uuid.uuid4())
            sli.sl[nel] = plt.Slider(sli.ax[nel], '$' + nel + '$',
                                     get_range(newval)[0],
                                     get_range(newval)[1],
                                     valinit=newval, facecolor='red')
            sli.ax[nel].set_visible(True)
            sli.sl[nel].on_changed(lambda vald, eld=nel:
                                   update_values(vald, 's', 'iin', eld))  
            sli.axtb[nel] = plt.axes([0.05 + nelind * 0.125 + 0.05, 0.1, 0.05,
                                      0.025], label=uuid.uuid4())
            sli.tb[nel] = plt.matplotlib.widgets.TextBox(sli.axtb[nel], '',
                                                         initial = '%g' %
                                                         newval)
            sli.axtb[nel].set_visible(True)
            sli.tb[nel].on_submit(lambda vald, eld=nel:
                                  update_values(vald, 'tb', 'iin', eld))
        t_start_box.on_submit(lambda vald: None)
        t_start_box.set_val(t_start)
        t_start_box.on_submit(update_t_start)
        t_end_box.on_submit(lambda vald: None)
        t_end_box.set_val(t_end)
        t_end_box.on_submit(update_t_end)
        atol_box.on_submit(lambda vald: None)
        atol_box.set_val(atol)
        atol_box.on_submit(update_atol)
        rtol_box.on_submit(lambda vald: None)
        rtol_box.set_val(rtol)
        rtol_box.on_submit(update_rtol)        
        plot_data(t, s, i_bias_ret, i_in_ret)


    def save_param_vals(val):
        '''Writes the AP and currents for the present set of parameter
        values to a text file.  Invoked by the 'W' button.
        inputs:  val:  (dummy variable needed for binding)
        output:  writes a text file called 'jj_ap.out' which contains
                 the AP and current data calculated for the present
                 set of parameter values.  Each row is
                 time, AP, i_bias, and i_in (where AP is the state
                 variable specified by index) in that order.
        errors:  none.'''

        (t, s, i_bias_ret, i_in_ret) = int_jj1(jjn, t_end=t_end, atol=atol,
                                               rtol=rtol)
        f = open('jj_ap.out', 'w')
        for j in range(len(t)):
            f.write('%g %g %g %g\n' % (t[j], s[j, index], i_bias_ret[j],
                                       i_in_ret[j]))
        f.close()


    def update_atol(text):
        '''Recalculates and replots the AP and current data when atol
        is changed.  Invoked by changes to the atol text box.
        inputs:  text:  the current contents of the atol text box.
        output:  recalculates and replots the AP and currents with the
                 new atol value.
        errors:  none.'''

        nonlocal atol

        atol = eval(text)
        update_plot(None)


    def update_plot(val):
        '''Recalculates and replots the AP and current data.  Invoked
        whenever a slider or text entry box is changed.
        inputs:  val:  (dummy variable needed for binding)
        output:  recalculates and replots the AP and current data.
        errors:  none.'''

        jjn.p = JJN_Param(**{el:sl.sl[el].val for el in JJN_Param_Names})
        jjn.i_bias.p = jjn.i_bias.Param(**{el:slb.sl[el].val
                                           for el in jjn.i_bias.Param_Names})
        jjn.i_in.p = jjn.i_in.Param(**{el:sli.sl[el].val
                                       for el in jjn.i_in.Param_Names})
        (t, s, i_bias_ret, i_in_ret) = int_jj1(jjn, t_end=t_end, atol=atol,
                                               rtol=rtol)
        plot_data(t, s, i_bias_ret, i_in_ret)

        
    def update_rtol(text):
        '''Recalculates and replots the AP and current data when rtol
        is changed.  Invoked by changes to the rtol text box.
        inputs:  text:  the current contents of the rtol text box.
        output:  recalculates and replots the AP and currents with the
                 new rtol value.
        errors:  none.'''

        nonlocal rtol
        
        rtol = eval(text)
        update_plot(None)


    def update_t_end(text):
        '''Recalculates and replots the AP and current data when t_end
        is changed.  Invoked by changes to the t_end text box.
        inputs:  text:  the current contents of the t_end text box.
        output:  recalculates and replots the AP and currents with the
                 new t_end value.
        errors:  none.'''
        
        nonlocal t_end

        t_end = eval(text)
        update_plot(None)


    def update_t_start(text):
        '''Recalculates and replots the AP and current data when t_start
        is changed.  Invoked by changes to the t_start text box.
        inputs:  text:  the current contents of the t_start text box.
        output:  recalculates and replots the AP and currents with the
                 new t_start value.
        errors:  none.'''

        nonlocal t_start

        t_start = eval(text)
        update_plot(None)


    def update_values(val, widtype, param_set, el):
        '''Called whenever a slider or text entry button value is changed;
        updates the value of the other member of the pair.
        inputs:  widtype:  's' for slider, 'tb' for text entry button
                 param_set:  'jjnp' for JJN circuit parameters, 'ibias'
                   and 'iin' respectively for bias and input current params
                 el:  name of the parameter
        output:  if a slider value has changed, updates the value of the
                 corresponding text entry button, and vice-versa.  Replots
                 the figure.
        errors:  none.'''
        
        nonlocal uv_primary
        
        if param_set == 'jjnp':
            if widtype == 's' and uv_primary:
                uv_primary = False
                sl.tb[el].set_val(float('%5.4f' % sl.sl[el].val))
                uv_primary = True
            elif widtype == 'tb' and uv_primary:
                uv_primary = False
                sl.sl[el].set_val(eval(sl.tb[el].text))
                uv_primary = True
        elif param_set == 'ibias':
            if widtype == 's' and uv_primary:
                uv_primary = False
                slb.tb[el].set_val(float('%5.4f' % slb.sl[el].val))
                uv_primary = True
            elif widtype == 'tb' and uv_primary:
                uv_primary = False
                slb.sl[el].set_val(eval(slb.tb[el].text))
                uv_primary = True
        elif param_set == 'iin':
            if widtype == 's' and uv_primary:
                uv_primary = False
                sli.tb[el].set_val(float('%5.4f' % sli.sl[el].val))
                uv_primary = True
            elif widtype == 'tb' and uv_primary:
                uv_primary = False
                sli.sl[el].set_val(eval(sli.tb[el].text))
                uv_primary = True                
        if uv_primary:
            update_plot(None)
            

    # main function code
    uv_primary = True
    fig = plt.figure()
    fig.set_size_inches(10,8)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_position([0.1,0.3,0.8,0.6])
    t_start = DEFAULT_T_START
    t_start_box = plt.matplotlib.widgets.TextBox(plt.axes([0.05, 0.25, 0.04,
                                                           0.02]),
                                                 '$t_{start}$',
                                                 initial='%g' % t_start)
    t_start_box.on_submit(update_t_start)    
    t_end = DEFAULT_T_END
    t_end_box = plt.matplotlib.widgets.TextBox(plt.axes([0.93, 0.25, 0.04,
                                                         0.02]),
                                               '$t_{end}$',
                                               initial='%g' % t_end)
    t_end_box.on_submit(update_t_end)
    atol_box = plt.matplotlib.widgets.TextBox(plt.axes([0.15, 0.25, 0.1,
                                                        0.02]),
                                              'atol',
                                              initial='%g' % DEFAULT_ATOL)
    atol = DEFAULT_ATOL
    atol_box.on_submit(update_atol)
    rtol_box = plt.matplotlib.widgets.TextBox(plt.axes([0.3, 0.25, 0.1, 0.02]),
                                              'rtol',
                                              initial='%g' % DEFAULT_RTOL)
    rtol = DEFAULT_RTOL
    rtol_box.on_submit(update_rtol)
    sl = SL_Param({}, {}, {}, {})
    for el in JJN_Param_Names:
        elind = JJN_Param_Names.index(el)
        sl.ax[el] = plt.axes([0.05 + elind * 0.2, 0.2, 0.05, 0.025])
        pval = jjn.p[elind]
        sl.sl[el] = plt.Slider(sl.ax[el], '$\\' + el + '$', get_range(pval)[0],
                               get_range(pval)[1], valinit=pval,
                               facecolor='blue')
        sl.sl[el].on_changed(lambda vald, eld=el:
                             update_values(vald, 's', 'jjnp', eld))  
        sl.axtb[el] = plt.axes([0.05 + elind * 0.2 + 0.05, 0.2, 0.05, 0.025],
                               label=uuid.uuid4())
        sl.tb[el] = plt.matplotlib.widgets.TextBox(sl.axtb[el], '',
                                                   initial = '%g' % pval)
        sl.tb[el].on_submit(lambda vald, eld=el:
                            update_values(vald, 'tb', 'jjnp', eld))
    slb = SL_Param({}, {}, {}, {})
    for el in jjn.i_bias.Param_Names:
        elind = jjn.i_bias.Param_Names.index(el)
        slb.ax[el] = plt.axes([0.05 + elind * 0.2, 0.15, 0.05, 0.025])
        pval = jjn.i_bias.p[elind]
        slb.sl[el] = plt.Slider(slb.ax[el], '$' + el + '$',
                                get_range(pval)[0], get_range(pval)[1],
                                valinit=pval, facecolor='green')
        slb.sl[el].on_changed(lambda vald, eld=el:
                              update_values(vald, 's', 'ibias', eld))
        slb.axtb[el] = plt.axes([0.05 + elind * 0.2 + 0.05, 0.15, 0.05, 0.025],
                                label=uuid.uuid4())
        slb.tb[el] = plt.matplotlib.widgets.TextBox(slb.axtb[el], '',
                                                    initial = '%g' % pval)
        slb.tb[el].on_submit(lambda vald, eld=el:
                             update_values(vald, 'tb', 'ibias', eld))
    sli = SL_Param({}, {}, {}, {})
    for el in jjn.i_in.Param_Names:
        elind = jjn.i_in.Param_Names.index(el)
        sli.ax[el] = plt.axes([0.05 + elind * 0.125, 0.1, 0.05, 0.025])
        pval = jjn.i_in.p[elind]
        sli.sl[el] = plt.Slider(sli.ax[el], '$' + el + '$',
                                get_range(pval)[0], get_range(pval)[1],
                                valinit=pval, facecolor='red')
        sli.sl[el].on_changed(lambda vald, eld=el:
                              update_values(vald, 's', 'iin', eld))
        sli.axtb[el] = plt.axes([0.05 + elind * 0.125 + 0.05, 0.1, 0.05,
                                 0.025], label=uuid.uuid4())
        sli.tb[el] = plt.matplotlib.widgets.TextBox(sli.axtb[el], '',
                                                    initial = '%g' % pval)
        sli.tb[el].on_submit(lambda vald, eld=el:
                             update_values(vald, 'tb', 'iin', eld))        
    ib = B_Param({}, {})
    for el in izh_param_list:
        elind = izh_param_list.index(el)
        ib.ax[el] = plt.axes([0.05 + elind * 0.03, 0.01, 0.03, 0.03])
        ib.b[el] = plt.Button(ib.ax[el], el)
        ib.b[el].on_clicked(lambda vald, el=el:
                            reset_params(vald, izh_param[el]))
    pvba = plt.axes([0.95, 0.06, 0.025, 0.05])
    pvb = plt.Button(pvba, '?')
    pvb.on_clicked(print_param_vals)
    rvba = plt.axes([0.925, 0.06, 0.025, 0.05])
    rvb = plt.Button(rvba, '!')
    rvb.on_clicked(lambda vald: reset_params(vald, (jjn, t_end)))
    svba = plt.axes([0.9, 0.06, 0.025, 0.05])
    svb = plt.Button(svba, 'W')
    svb.on_clicked(save_param_vals)
    i4ba = plt.axes([0.875, 0.06, 0.025, 0.05])
    i4b = plt.Button(i4ba, r'$\Phi$')
    i4b.on_clicked(lambda vald: change_index(vald, 4))
    i3ba = plt.axes([0.95, 0.11, 0.025, 0.05])
    i3b = plt.Button(i3ba, r"$\dot{\phi}_c$")
    i3b.on_clicked(lambda vald: change_index(vald, 3))
    i2ba = plt.axes([0.925, 0.11, 0.025, 0.05])
    i2b = plt.Button(i2ba, r"$\phi_c$")
    i2b.on_clicked(lambda vald: change_index(vald, 2))
    i1ba = plt.axes([0.9, 0.11, 0.025, 0.05])
    i1b = plt.Button(i1ba, r"$\dot{\phi}_p$")
    i1b.on_clicked(lambda vald: change_index(vald, 1))
    i0ba = plt.axes([0.875, 0.11, 0.025, 0.05])
    i0b = plt.Button(i0ba, r"$\phi_p$")
    i0b.on_clicked(lambda vald: change_index(vald, 0))
    helpba = plt.axes([0.925, 0.01, 0.05, 0.05])
    helpb = plt.Button(helpba, 'Help')
    helpb.on_clicked(lambda vald: display_help_window(vald))
    (t, s, i_bias_ret, i_in_ret) = int_jj1(jjn, t_end=t_end)
    plt.sca(ax)
    p, = plt.plot(t, s[:, index], color='blue')
    ax.set_xlabel(r"time [a. u.]")
    if index == 0:
        ylabel = r"$\phi_p$ \, [a. u.]"
    elif index == 1:
        ylabel = r"$\dot{\phi}_p$ \, [a. u.]"
    elif index == 2:
        ylabel = r"$\phi_c$ \, [a. u.]"
    elif index == 3:
        ylabel = r"$\dot{\phi}_c$ \, [a. u.]"
    elif index == 4:
        ylabel = r"$\Phi$ \, [a. u.]"
    ax.set_ylabel(ylabel, color='blue')
    (yl, yu) = y_plot_range(s[:, index], 'ap')
    ax.set_xlim(0, t_end)
    ax.set_ylim(yl, yu)    
    ax2 = ax.twinx()
    ax2.set_ylabel("$i_{in}$ \, [a. u.]", color='red')
    plt.sca(ax2)
    ax2.set_position([0.1,0.3,0.8,0.6])    
    p2, = plt.plot(t, i_in_ret, color='red')
    (y2l, y2u) = y_plot_range(i_in_ret, 'i_in')
    ax2.set_xlim(0, t_end)
    ax2.set_ylim(y2l, y2u)
    ax3 = ax.twinx()
    ax3.set_ylabel(".\n$i_{bias}$ \, [a. u.]", color='green')
    plt.sca(ax3)
    ax3.set_position([0.1,0.3,0.8,0.6])    
    p3, = plt.plot(t, i_bias_ret, color='green')
    ax3.set_xlim(0, t_end)
    ax3.set_ylim(y2l, y2u)
    plt.show()



def plot_to_file(jjn=JJN(), fname_base='jj_ap', t_start=0, t_end=2000, index=1,
                 atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL, write_data=True,
                 write_fig=False, plot_title=None):
    '''Calculates and writes JJN AP and current data to a data file and/or
    a figure file.
    inputs:  jjn:  a JJN object.
             [fname_base]:  the base name of the file(s) to be generated.
               The extension '.dat' is appended for data files, and the
               extension(s) specified by write_fig for figure files.
               (Default:  'jj_ap')
             [t_start]:  start time for plots (which show range t_start to
               t_end).  (Default:  0)
             [t_end]:  end time for plots / total integration time for data.
               (Default:  2000)
             [index]:  index of the state vector to plot.  (Default:  1,
               phi_p')
             [atol]:  absolute tolerance for ODE integrator
               (Default:  DEFAULT_ATOL)
             [rtol]:  relative tolerance for ODE integrator
               (Default:  DEFAULT_RTOL)
             [write_data]:  if True, writes data to a text file in the
                 current directory which contains the AP and current data 
                 calculated for the present set of parameter values.  The
                 values in each row are time, phi_p, phi_p', phi_c, phi_c', 
                 flux, i_bias, and i_in (all at at that time value), in that 
                 order.  The name of the file is fname_base + '.dat'.  
                 (Default:  True)
             [write_fig]:  variable of different types which contains
                 information about the figure.  If write_fig=False, no figure 
                 file is written.  If write_fig is one of the strings in
                 the global constant SUPPORTED_IMAGE_FILE_FORMATS, a file is
                 written in that format showing the AP and current data in the 
                 same way as mess_around().  If write_fig is a tuple, the first
                 element should be a string for the image file format, and the
                 remaining ones elements of SUPPORTED_IMAGE_FILE_CMPENC; the 
                 file will then be processed according to the encodings and/or
                 compressions thus specified, in that order.  The only 
                 currently supported compression algorithm is LZMA, specified 
                 by 'xz', and the only encoding algorithm is Base64, 'b64'.  
                 The filename is fname_base + '.eps'/'.jpg'/etc + the 
                 extensions for the compression and/or encoding; thus
                 if write_fig=('eps','xz','b64'), the file's name is
                 fname_base + '.eps' + '.xz' + '.b64', and it will be an
                 EPS figure which is xz-compressed and then Base64-encoded.  
                 Finally, if the final element of the tuple is 'ram', then
                 the figure file is not written to disk; instead, 
                 plot_to_file() returns a tuple whose first element is the 
                 figure file name and the second of which is a bytestring 
                 containing the *figure file data* with the format, 
                 compression, and encodings specified.  Thus if 
                 write_fig=('eps','xz','b64','ram'), nothing is written to
                 disk, and a tuple 
                 (fname_base + '.eps' + '.xz' + '.b64', [image data]) is
                 returned.  (Default:  False)
             [plot_title]: string for plot title; if None, none.  (Default:
                 False)
    output:  calculates data and writes file(s) as specified above.  Returns
             True unless write_fig is True; if write_fig is a tuple whose last 
             element is 'ram', returns a tuple with the figure file name and 
             image data as described above; otherwise, returns the figure file
             name as a string.
    errors:  prints error and returns False if invalid variable values or
             any other errors.'''
    
    (t, s, i_bias_ret, i_in_ret) = int_jj1(jjn, t_end=t_end, atol=atol,
                                           rtol=rtol)
    retval = True
    if write_data:
        f = open(fname_base + '.dat', 'w')
        for j in range(len(t)):
            f.write('%g %g %g %g %g %g %g %g\n' % (t[j], s[j, 0], s[j, 1],
                                                   s[j, 2], s[j, 3], s[j, 4],
                                                   i_bias_ret[j], i_in_ret[j]))
        f.close()
    if write_fig:
        write_to_ram = False
        if type(write_fig) is str:
            fig_type = write_fig
            fig_filter = ()
        elif type(write_fig) is tuple:
            fig_type = write_fig[0]
            if write_fig[-1] == 'ram':
                write_to_ram = True
                fig_filter = write_fig[1:-1]
            else:
                fig_filter = write_fig[1:]
        else:
            print('error:  figure file specification not valid')
            return False
        if fig_type not in SUPPORTED_IMAGE_FILE_FORMATS:
            print("error:  supported figure file formats:")
            print("        %s" % ', '.join(SUPPORTED_IMAGE_FILE_FORMATS))
            return False
        for el in fig_filter:
            if el not in SUPPORTED_IMAGE_FILE_CMPENC:
                print("error:  supported figure file compression/encoding "
                      "algorithms are:")
                print("    %s" % ', '.join(SUPPORTED_IMAGE_FILE_CMPENC))
                return False
        fig = plt.figure()
        fig.set_size_inches(5, 4)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_position([0.15, 0.15, 0.7, 0.7])
        plt.sca(ax)
        if t_start > 0:
            if t_start > t_end:
                print("error:  start time must be <= end time")
                return False
            j_start = 0
            while t[j_start] < t_start:
                j_start = j_start + 1
        else:
            j_start = 0    
        (yl, yu) = y_plot_range(s[j_start:, index], 'ap')
        (y2l, y2u) = y_plot_range(i_in_ret[j_start:], 'i_in')
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(yl, yu)
        p, = plt.plot(t[j_start:], s[j_start:, index], color='blue')
        ax.set_xlabel(r"time [a. u.]")
        if index == 0:
            ylabel = r"$\phi_p$ [a. u.]"
        elif index == 1:
            ylabel = r"$\dot{\phi}_p$ [a. u.]"
        elif index == 2:
            ylabel = r"$\phi_c$ [a. u.]"
        elif index == 3:
            ylabel = r"$\dot{\phi}_c$ [a. u.]"
        elif index == 4:
            ylabel = r"$\Phi$ \, [a. u.]"
        ax.set_ylabel(ylabel, color='blue')
        ax2 = ax.twinx()
        ax2.set_ylabel(r"$i_{in}$ [a. u.]", color='red')
        plt.sca(ax2)
        ax2.set_position([0.15, 0.15, 0.7, 0.7])
        p2, = plt.plot(t[j_start:], i_in_ret[j_start:], color='red')
        ax2.set_xlim(t_start, t_end)
        ax2.set_ylim(y2l, y2u)
        ax3 = ax.twinx()
        ax3.set_ylabel("\x20\n$i_{bias}$ [a. u.]", color='green')
        plt.sca(ax3)
        ax3.set_position([0.15, 0.15, 0.7, 0.7])
        p3, = plt.plot(t[j_start:], i_bias_ret[j_start:], color='green')
        ax3.set_xlim(t_start, t_end)
        ax3.set_ylim(y2l, y2u)
        ramfile = io.BytesIO()
        if plot_title:
            plt.title(plot_title)
        plt.savefig(ramfile, format=fig_type)
        plt.close()
        bfig = ramfile.getbuffer().tobytes()
        for ce in fig_filter:
            if ce == 'xz':
                bfig = lzma.compress(bfig)
            elif ce == 'b64':
                bfig = base64.b64encode(bfig)
        fname = fname_base + '.' + fig_type
        for ce in fig_filter:
            fname = fname + '.' + ce
        if write_to_ram:
            retval = (fname, bfig)
        else:
            retval = fname
            f = open(fname, 'wb')
            f.write(bfig)
            f.close()
    return retval



def vary_params(jjn, base_label='', vary_frac=0.1, N=100, t_end=1000,
                print_progress=True, plot_title=False):
    '''Randomly varies parameter values within specified ranges, calculates
    JJN APs, and writes them to figure files.
    inputs:  jjn:  a JJN object.
             [base_label]:  string used as prefix for file names and
               graph titles.
             [vary_frac]:  float or dict specifying the range over which
               to vary parameter values.  If a float, *all* parameters, both
               for the JJN and for the bias and input currents, are varied
               by this percentage from their base values in jjn.  If
               a dict, the the keys are the JJN parameters (like 'Gamma'), 
               'i_bias', or 'i_in'.  For JJN parameters, the 'value' is
               either a single float, interpreted as a percentage like 
               before (though only for that parameter), or a tuple giving the 
               minimum and maximum values of that parameter.  For the
               'i_bias' and 'i_in' keys, the 'value' is a subdict whose
               keys are the parameters for that current object, with the
               'values' for those as for the JJN parameters.  (Default:  0.1)
             [N]:  total number of randomized plots to generate (Default:  100)
             [t_end]:  end time for plots, which start at 0.  If this is a
               tuple, the first number is the start time.  (Default:  1000)
             [print_progress]:  if True, will print progress as the plots
               are being generated.  (Default:  True)
             [plot_title]:  If True, the values of parameters that are
               *different* from the base ones in jjn will be shown in the
               figure titles.  (Default:  False)
    output:  random values for the JJN and current parameters are chosen
             within the ranges specified by vary_frac.  If a parameter
             does not appear in vary_frac, it is kept constant at its
             value in jjn, unless vary_frac is itself a float.  The JJN AP
             is then calculated and written to an EPS file whose name is a
             (lengthy) list of all the parameters and their values, with
             base_label as the prefix.  Returns a list of the names of 
             the figure files generated.
    errors:  no error checking, so be careful to use valid values.'''

    fname_list = []
    param_base = jjn.p
    ib_param_base = jjn.i_bias.p
    iin_param_base = jjn.i_in.p
    param_range = {}
    ib_param_range = {}
    iin_param_range = {}
    if type(vary_frac) is float:
        for param in JJN_Param_Names:
            param_val = param_base._asdict()[param]
            param_range[param] = (param_val * (1 - vary_frac),
                                  param_val * (1 + vary_frac))
        for param in jjn.i_bias.Param_Names:
            param_val = ib_param_base._asdict()[param]
            ib_param_range[param] = (param_val * (1 - vary_frac),
                                     param_val * (1 + vary_frac))
        for param in jjn.i_in.Param_Names:
            param_val = iin_param_base._asdict()[param]
            iin_param_range[param] = (param_val * (1 - vary_frac),
                                      param_val * (1 + vary_frac))
    elif type(vary_frac) is dict:
        for param in vary_frac.keys():
            if param not in ['i_bias', 'i_in']:
                if type(vary_frac[param]) is float:
                    param_val = param_base._asdict()[param]
                    param_range[param] = (param_val * (1 - vary_frac[param]),
                                          param_val * (1 + vary_frac[param]))
                else:
                    param_range[param] = vary_frac[param]
            else:
                for iparam in vary_frac[param].keys():
                    if param == 'i_bias':
                        if type(vary_frac[param][iparam]) is float:
                            ib_param_val = ib_param_base._asdict()[iparam]
                            ibvf = vary_frac[param][iparam]
                            ib_param_range[iparam] = ((ib_param_val *
                                                       (1 - ibvf),
                                                       ib_param_val *
                                                       (1 + ibvf)))
                        else:
                            ib_param_range[iparam] = vary_frac[param][iparam]
                    elif param == 'i_in':
                        if type(vary_frac[param][iparam]) is float:
                            iin_param_val = iin_param_base._asdict()[iparam]
                            iinvf = vary_frac[param][iparam]
                            iin_param_range[iparam] = ((iin_param_val *
                                                        (1 - iinvf),
                                                        iin_param_val *
                                                        (1 + iinvf)))
                        else:
                            iin_param_range[iparam] = vary_frac[param][iparam]
    for n in range(N):
        point = param_base._asdict()
        ib_point = ib_param_base._asdict()
        iin_point = iin_param_base._asdict()
        ptitle=base_label + '.'
        for param in param_range.keys():
            param_point_val = (param_range[param][0] + random.random() *
                               (param_range[param][1] - param_range[param][0]))
            point[param] = param_point_val
            ptitle = ptitle + '%s=%g,' % (param, param_point_val)
        for ib_param in ib_param_range.keys():
            ib_param_point_val = (ib_param_range[ib_param][0] +
                                  (random.random() *
                                   (ib_param_range[ib_param][1] -
                                    ib_param_range[ib_param][0])))
            ib_point[ib_param] = ib_param_point_val
            ptitle = ptitle + '%s=%g,' % (ib_param, ib_param_point_val)
        for iin_param in iin_param_range.keys():
            iin_param_point_val = (iin_param_range[iin_param][0] +
                                   (random.random() *
                                    (iin_param_range[iin_param][1] -
                                     iin_param_range[iin_param][0])))
            iin_point[iin_param] = iin_param_point_val
            ptitle = ptitle + '%s=%g,' % (iin_param, iin_param_point_val)
        fname_base = base_label + '.'
        for param in point.keys():
            fname_base = fname_base + '%s=%g,' % (param, point[param])
        for iparam in ib_point.keys():
            fname_base = fname_base + 'ib.%s=%g,' % (iparam, ib_point[iparam])
        for iparam in iin_point.keys():
            fname_base = fname_base + 'iin.%s=%g,' % (iparam,
                                                      iin_point[iparam])
        new_jjn = JJN(JJN_Param(**point), i_bias=type(jjn.i_bias)(**ib_point),
                      i_in=type(jjn.i_in)(**iin_point))
        if type(t_end) is tuple:
            tstart = t_end[0]
            tend = t_end[1]
        else:
            tstart = 0
            tend = t_end
        if plot_title:
            new_fname = plot_to_file(jjn=new_jjn, fname_base=fname_base,
                                     t_start = tstart, t_end=tend,
                                     write_data=False, write_fig=('eps',),
                                     plot_title=ptitle)
        else:
            new_fname = plot_to_file(jjn=new_jjn, fname_base=fname_base,
                                     t_start = tstart, t_end=tend,
                                     write_data=False, write_fig=('eps',))
        fname_list = fname_list + [new_fname]
        if print_progress:
            print('.',end='',flush=True)
            if not ((n + 1) % 10):
                print('(%g)' % (n + 1), end='', flush=True)
    if print_progress:
        print('')
    return fname_list



def y_plot_range(y, dtype='ap'):
    '''Calculates a suitable y-axis range for a plot.
    inputs:  y:  an ordered array of values, corresponding to the vertical
               axis of a plot.
             [dtype]:  the type of data in y:  'ap':  action potential;
               'i_in':  input current.  (Default:  'ap')
    output:  returns (yl, yu), where these are the lower and upper
               limits of the y-axis range.  For dtype 'ap', the axis
               range is the minimum value in y minus 1/10 of the total
               y value range to the maximum value plus 1/10 of the total
               range.  For example, if the minimum value of y is 1 and
               the maximum is 11, then (yl, yu) = (0.9, 11.1).  If y
               is constant and nonzero, then the axis range is 0.1 above
               and below its value.  If y is constant and 0, the axis
               range is -1 to 1.  For dtype 'i_in', the calculation is
               similar, except it is based on the largest postive
               value of y and goes from -0.2 time to 4.0 times that value.
    errors:  if dtype is unrecognized, returns (None, None).'''
    
    miny = min(y)
    maxy = max(y)
    yrange = maxy - miny
    if dtype == 'ap':
        if yrange > 0:
            (yl, yu) = (miny - 0.1 * yrange, maxy + 0.1 * yrange)
        else:
            if miny != 0:
                (yl, yu) = sorted((0.9 * miny, 1.1 * miny))
            else:
                (yl, yu) = (-1, 1)
    elif dtype == 'i_in':
        if yrange > 0:
            #ynz = [el for el in y if el != 0]
            #ystep = ynz[0]
            #(yl, yu) = sorted((-0.2 * ystep, 4.0 * ystep))
            (yl, yu) = (miny - 0.1 * yrange, maxy + 0.1 * yrange)
        else:
            if miny != 0:
                (yl, yu) = sorted((-0.2 * miny, 4.0 * miny))
            else:
                (yl, yu) = (-0.2, 4.0)
    else:
        (yl, yu) = (None, None)
    return (yl, yu)       
# ---------------------------------------------------------------------|----- #




# ----------------------------------- MAIN ----------------------------|----- #
if __name__ == '__main__':
    mess_around()
# =====================================================================|===== #
