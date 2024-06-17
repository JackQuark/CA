# package for CA, 2024
# dataset: VVM simulation, tpe20110802cln

# include the following Class
# Location: select the location by ID or name
# (download the .nc file for the spot and spot_info.csv first)
# Thermo: thermodynamic functions

# by Quark 2024/5 weeeeeeeeeeeeeeeeeeeeeeeeeeeeeek 2 
# ==================================================
import os
import sys
import time
import netCDF4              as nc
import numpy                as np
import matplotlib.pyplot    as plt
from scipy                  import integrate
# ==================================================
# location selecte
class Location:

    def __init__(self, filepath=None, **kwargs):

        # id select
        if 'id' in kwargs:
            self.load()
            
            if kwargs['id'] in self.id:
                idx = int(np.where(self.id == kwargs['id'])[0])
                self.loc = [self.lon[idx], self.lat[idx]]
                self.file_name = self.id[idx]
            else:
                raise ValueError('Invalid ID')
        
        # 
        elif 'name' in kwargs:
            self.cache: dict = {
                    'Quark': [120.32586, 23.96824],
                    'Yushan': [120.95952, 23.48761],
                    'Sparrow': [121.01422, 24.82785],
                    'Taipei': [121.51485, 25.03766],
                    'Vincent': [121.61328, 23.97513],
                    'Gary': [121.62970, 24.14743],
                    'Cubee': [121.74103, 24.26676 ]
                    }
            
            if kwargs['name'] in self.cache:
                self.loc = self.cache[kwargs['name']]
                self.file_name = kwargs['name']
            else:
                raise ValueError('Invalid cache name')
            
        else:
            self.loc = [120.32586, 23.96824]

        self.file = f'{filepath}\\{self.loc[0]:.5f}_{self.loc[1]:.5f}.nc'
            
    def load(self):
        self.id = \
            np.loadtxt('..\spots_info.csv', dtype=str,
                       delimiter=',', skiprows=1, usecols=(0), unpack=True)
            
        self.lon, self.lat, self.lu = \
            np.loadtxt('..\spots_info.csv', dtype=float,
                       delimiter=',', skiprows=1, usecols=(2,3,5), unpack=True)

# ==================================================
# thermo functioms
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
    def T_2_Tv(self, Temp, qv):
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
    
# ==================================================
# main
def main():
    return None

# ==================================================
# execute
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))
# ==================================================