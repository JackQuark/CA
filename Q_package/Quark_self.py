# Useless code
# 
# By Quark
# ==================================================
import os
import time
import netCDF4              as nc
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gs
import sympy                as sym
# from sympy.solvers          import solve
# from sympy                  import Symbol
# from scipy                  import integrate
# ==================================================

# AS
class AS:
    def __init__(self):
        pass
        
    class Thermo:
        
        def __init__(self):
            # thermodynamic parameters
            self.Rd = 287. # J/kg/K
            self.Cp = 1004. # J/kg/K
            self.Lv = 2.5e6 # J/kg
            self.Rv = 461. # J/kg/K
            self.g  = 9.81 # m/s^2
            self.epsilon= self.Rd / self.Rv
            
            self.Gamma_d = self.g / self.Cp # K/m
        
        # potential Temp. [K] <=> Temp. [K]
        def th_2_T(self, th, P):
            Temp = th * (P / 1e3)**(self.Rd/self.Cp)
            return Temp
        
        def T_2_th(self, Temp, P):
            th = Temp * (1e3 / P)**(self.Rd/self.Cp)
            return th
        
        # virtual Temp. [K]
        def T_2_Tv(self, Temp, qv, rho):
            Tv = Temp * (1 + qv * (1-self.epsilon) / self.epsilon)
            return Tv

        # equivalent potential Temp. [K]
        def theta_e(self, Temp, qv, p):
            return self.T_2_th(Temp + qv * self.Lv / self.Cp, p)
        
        def theta_e_ac(self, th, Tc, qv):
            return th * np.exp((self.Lv * qv) / (self.Cp * Tc)) 
            
        # cc equation
        # Temp. [K] <=> saturated vapor pressure [hPa]
        def cc_equation(self, Temp):
            es = 6.112 * np.exp(self.Lv/self.Rv * (1/273 - 1/Temp)) # hPa
            return es
        
        def anti_cc_equation(self, es):
            Temp = 1 / ((1/273.15) - (self.Rv/self.Lv) * np.log(es/6.112))
            return Temp
            
        # specific humidity [kg/kg] <=> vapor pressure [hPa]
        def e_2_qv(self, e, P):
            qv = (self.Rd/self.Rv) * e / P
            return qv
        
        def qv_2_e(self, qv, P):
            e = P * qv / (self.Rd/self.Rv)
            return e
        
        # moist adiabatic lapse rate [K/m]
        def gamma_m(self, T, qvs):
            Cp_star = (self.Cp *
                (1 + (self.Lv**2 * qvs / (self.Cp * self.Rv * T**2))) /
                (1 + (self.Lv * qvs / (self.Rd * T))))
            return self.g / Cp_star
        
    def obj_thermo(self):
        return self.Thermo()
        
# Matplotlib
class Draw:
    
    @ staticmethod
    def autoticks(Min, Max, interval, n=6, 
                  side='left', start=None):
            
        if side == 'right':
            go, end = np.floor(Max), Min
            interval = -interval
            
        else:
            go, end = np.ceil(Min), Max
        
        step = 0
        tmp  = n + 1
        while tmp > n:
            step += interval
            arr   = np.arange(go, end, step)
            tmp   = len(arr)
        
        return arr

# Mathematical / Numerical
class Math:
    def __init__(self):
        pass