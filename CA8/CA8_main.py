# CA7 / mixed layer model and VVM
# ==================================================
import os
import sys
import time
import netCDF4              as nc
import numpy                as np
import matplotlib.pyplot    as plt
from scipy                  import integrate
from matplotlib.animation   import FuncAnimation, FFMpegWriter
from matplotlib.gridspec    import GridSpec

from metpy.units            import units
from metpy.calc             import dewpoint, cape_cin,\
    dewpoint_from_relative_humidity, dewpoint_from_specific_humidity

sys.path.append('..')
from Q_package.CA_ import Thermo, Location
from Q_package.Quark_self import Draw as Q_draw
# ==================================================
# VVM data
class VVM:
    
    def __init__(self, file: str):
        
        with nc.Dataset(file) as rootgrp:

            topo: int = rootgrp.variables['topo'][0]
            self.m_zc = rootgrp.variables['m_zc'][topo:]
            self.p_zc = rootgrp.variables['p_zc'][topo:]
            self.rho  = rootgrp.variables['rho' ][topo:]
            self.th   = rootgrp.variables['th'  ][:,topo:]
            self.qv   = rootgrp.variables['qv'  ][:,topo:]
            self.qc   = rootgrp.variables['qc'  ][:,topo:]
            self.qr   = rootgrp.variables['qr'  ][:,topo:]
            self.qi   = rootgrp.variables['qi'  ][:,topo:]
            self.wth  = rootgrp.variables['wth' ][:]
            self.wqv  = rootgrp.variables['wqv' ][:]
            
        self.Qth  = integrate.cumtrapz(self.wth, dx=600., initial=0) / self.rho[0]
        self.Qqv  = integrate.cumtrapz(self.wqv, dx=600., initial=0) / self.rho[0]
        self.temp = thermo.th_2_T(self.th, self.p_zc)
        self.es   = thermo.cc_equation(self.temp)
        self.qvs  = thermo.e_2_qv(self.es, self.p_zc)
        self.SH_flux = self.wth * thermo.Cp * (self.p_zc[0] / 1e3)**(0.286)
        self.LH_flux = self.wqv * thermo.Lv * (self.p_zc[0] / 1e3)**(0.286)
        
        self.qt    = np.zeros((145, len(self.m_zc)))
        self.th_e  = np.zeros((145, len(self.m_zc)))
        self.th_es = np.zeros((145, len(self.m_zc)))
        for t in range(145):
            self.qt[t,:]    = self.qv[t,:] + self.qc[t,:]
            self.th_e[t,:]  = thermo.theta_e(self.temp[t,:], self.qv[t,:], self.p_zc[:])
            self.th_es[t,:] = thermo.theta_e(self.temp[t,:], self.qvs[t,:], self.p_zc[:])
            
# ==================================================
# Mixed data
class Mix:

    def __init__(self):
        self.mix_layer(vvm.Qth, vvm.th, vvm.Qqv, vvm.qv, vvm.m_zc, vvm.p_zc)
        self.insert2_profile(vvm.th, vvm.qv, vvm.m_zc, vvm.p_zc)
        
        self.es    = thermo.cc_equation(self.temp_2)
        self.qvs   = thermo.e_2_qv(self.es, self.p_zc_2)
        self.es_d  = thermo.cc_equation(self.temp_d_2)
        self.qvs_d = thermo.e_2_qv(self.es_d, self.p_zc_2)
        
        self.Zc = np.zeros(145)
        for t in range(145):
            lev = len(np.where(self.qvm[t] < self.qvs_d[t])[0]) - 1
            
            dqv = self.qvm[t] - self.qvs_d[t, lev]
            
            if (self.m_zc_2[t, lev+1] - self.m_zc_2[t, lev]) == 0:
                dz = 0.
            else:
                dz = dqv * (self.m_zc_2[t, lev+1] - self.m_zc_2[t, lev]) / (self.qvs_d[t, lev+1] - self.qvs_d[t, lev])
            
            self.Zc[t] = self.m_zc_2[t, lev] + dz
        
        self.insert3_profile(vvm.th, vvm.qv, vvm.m_zc, vvm.p_zc)
        
    def mix_layer(self, Qth, th, Qqv, qv, m_zc, p_zc):
        
        self.relate_zc = m_zc - m_zc[0]
        self.slope_p  = (p_zc[1:] - p_zc[:-1]) / (m_zc[1:] - m_zc[:-1])
        self.slope_th = (m_zc[1:] - m_zc[:-1]) / (th[0, 1:] - th[0, :-1])
        self.slope_qv = (m_zc[1:] - m_zc[:-1]) / (qv[0, 1:] - qv[0, :-1])
        self.z_th_ = integrate.cumtrapz(self.relate_zc, th[0, :], initial=0)
        self.z_qv_ = integrate.cumtrapz(qv[0, :], self.relate_zc, initial=0)
        # index which mixed layer should insert
        self.lev = np.searchsorted(self.z_th_, Qth, side='right')

        # mix_H = sqrt(2 * Area * slope + z^2) + m_zc[0]
        self.Zm     = np.where(self.lev == 0, m_zc[0],
                      np.sqrt(2 * (Qth - self.z_th_[self.lev-1]) 
                      * self.slope_th[self.lev-1] 
                      + self.relate_zc[self.lev-1]**2) 
                      + m_zc[0])
        
        # zm count from surface (m_zc[topo] m)
        self.Zm_relate = self.Zm - m_zc[0]
        
        # distance between zm and previous level
        self.Zmdz   = self.Zm - m_zc[self.lev-1]
        
        # mix_th = previous level th + Zmdz / slope
        self.th_m   = np.where(self.Zm_relate == 0., th[0, 0],
                      th[0, self.lev-1] + 
                      self.Zmdz / self.slope_th[self.lev-1])

        # VVM_qv at Zm = previous level th + Zmdz / slope
        self.qv_Zm  = np.where(self.Zm_relate == 0., qv[0, 0], 
                      qv[0, self.lev-1] + 
                      self.Zmdz / self.slope_qv[self.lev-1])
        
        # mix_qv = (vvm_qv under zm + total wqv at that time) / mix_H
        self.qvm   = np.where(self.Zm_relate == 0., qv[0, 0],
                      (Qqv + self.z_qv_[self.lev-1] + 
                      (qv[0, self.lev-1] + self.qv_Zm) * self.Zmdz / 2)
                      / self.Zm_relate)

    # insert 2 points at Zm
    def insert2_profile(self, th, qv, m_zc, p_zc):
        level = np.where(self.lev == 0, 1, self.lev)

        length_2 = len(m_zc) + 2
        
        self.p_zc_2   = np.zeros((145, length_2))
        self.m_zc_2   = np.zeros((145, length_2))
        self.th_2     = np.zeros((145, length_2))
        self.qv_2     = np.zeros((145, length_2))
        self.temp_2   = np.zeros((145, length_2))
        self.temp_d_2 = np.zeros((145, length_2))
        
        for i in range(145): # [time, level+2]
            self.m_zc_2[i, :level[i]]           = m_zc[:level[i]]
            self.m_zc_2[i, level[i]:level[i]+2] = self.Zm[i]
            self.m_zc_2[i, level[i]+2:]         = m_zc[level[i]:]
            
            self.p_zc_2[i, :level[i]]           = p_zc[:level[i]]
            self.p_zc_2[i, level[i]:level[i]+2] = p_zc[level[i]-1] + (self.slope_p[level[i]-1]
                                              * (self.Zm[i] - m_zc[level[i]-1]))
            self.p_zc_2[i, level[i]+2:]         = p_zc[level[i]:]
            
            self.th_2[i, :level[i]+2] = self.th_m[i]
            self.th_2[i, level[i]+2:] = th[0, level[i]:]
            
            self.qv_2[i, :level[i]+1] = self.qvm[i]
            self.qv_2[i, level[i]+1]  = self.qv_Zm[i]
            self.qv_2[i, level[i]+2:] = qv[0, level[i]:]
            
            self.temp_2[i, :]   = thermo.th_2_T(self.th_2[i, :], self.p_zc_2[i, :])
            
            self.temp_d_2[i, :] = thermo.th_2_T(self.th_m[i], self.p_zc_2[i, :])
        
    # insert 2 points at Zm, 1 point at Zc
    def insert3_profile(self, th, qv, m_zc, p_zc):
        lev_Zm = np.where(self.lev == 0, 1, self.lev)
        lev_Zc = np.searchsorted(m_zc, self.Zc, side='right')
        length_3 = len(m_zc) + 3
        
        self.p_zc_3   = np.zeros((145, length_3))
        self.m_zc_3   = np.zeros((145, length_3))
        self.th_3     = np.zeros((145, length_3))
        self.qv_3     = np.zeros((145, length_3))
        self.qvs_3    = np.zeros((145, length_3))
        self.temp_3   = np.zeros((145, length_3))
        self.qc_3     = np.zeros((145, length_3))
        
        # for ideal parcial
        self.temp_d_3   = np.zeros((145, length_3))
        self.temp_parc  = np.zeros((145, length_3))
        self.qv_parc    = np.zeros((145, length_3))
        self.qvs_parc   = np.zeros((145, length_3))
        self.th_e_parc  = np.zeros((145, length_3))
        self.th_es_parc = np.zeros((145, length_3))
        
        self.th_e     = np.zeros((145, length_3))
        self.th_es    = np.zeros((145, length_3))
        
        for t in range(145): # [time, level+3]
            
            def moist_adiabatic(lev):
                tmp_H = self.m_zc_3[t, lev-1]
                tmp_p = self.p_zc_3[t, lev-1]
                tmp_temp = self.temp_parc[t, lev-1]
                tmp_qvs  = self.qvs_parc[t, lev-1]
                # index in vvm level
                tmp_lev  = np.argmin(np.abs(vvm.m_zc - tmp_H))
                step = 100
                
                # moist_adiabatic iterating
                while tmp_H < self.m_zc_3[t, lev]:
                    # if +100 too much
                    if tmp_H + step > self.m_zc_3[t, lev]:
                        step = self.m_zc_3[t, lev] - tmp_H
                    
                    tmp_H += step
                    tmp_p += step * self.slope_p[tmp_lev]
                    tmp_gamma = thermo.gamma_m(tmp_temp, tmp_qvs)
                    tmp_temp -= step * tmp_gamma
                    tmp_qvs   = thermo.e_2_qv(thermo.cc_equation(tmp_temp), tmp_p)
                
                return tmp_temp, tmp_qvs
                
            if self.Zm[t] < self.Zc[t]:
                # ========== env ==========
                self.m_zc_3[t, :lev_Zm[t]]              = m_zc[:lev_Zm[t]]
                self.m_zc_3[t, lev_Zm[t]:lev_Zm[t]+2]   = self.Zm[t]
                self.m_zc_3[t, lev_Zm[t]+2:lev_Zc[t]+2] = m_zc[lev_Zm[t]:lev_Zc[t]]
                self.m_zc_3[t, lev_Zc[t]+2]             = self.Zc[t]
                self.m_zc_3[t, lev_Zc[t]+3:]            = m_zc[lev_Zc[t]:]
                
                self.p_zc_3[t, :lev_Zm[t]]              = p_zc[:lev_Zm[t]]
                self.p_zc_3[t, lev_Zm[t]:lev_Zm[t]+2]   = p_zc[lev_Zm[t]-1] + ((self.Zm[t] - m_zc[lev_Zm[t]-1])
                                                                                * self.slope_p[lev_Zm[t]-1])
                self.p_zc_3[t, lev_Zm[t]+2:lev_Zc[t]+2] = p_zc[lev_Zm[t]:lev_Zc[t]]
                self.p_zc_3[t, lev_Zc[t]+2]             = p_zc[lev_Zc[t]-1] + ((self.Zc[t] - m_zc[lev_Zc[t]-1])
                                                                               * self.slope_p[lev_Zc[t]-1])
                self.p_zc_3[t, lev_Zc[t]+3:]            = p_zc[lev_Zc[t]:]
                
                self.th_3[t, :lev_Zm[t]+2]              = self.th_m[t]
                self.th_3[t, lev_Zm[t]+2:lev_Zc[t]+2]   = th[0, lev_Zm[t]:lev_Zc[t]]
                self.th_3[t, lev_Zc[t]+2]               = th[0, lev_Zc[t]-1] + ((self.Zc[t] - m_zc[lev_Zc[t]-1])
                                                                                / self.slope_th[lev_Zc[t]-1])
                self.th_3[t, lev_Zc[t]+3:]              = th[0, lev_Zc[t]:]
                
                self.temp_3[t, :]                       = thermo.th_2_T(self.th_3[t, :], self.p_zc_3[t, :])
                
                self.qv_3[t, :lev_Zc[t]+2]              = self.qv_2[t, :lev_Zc[t]+2]
                self.qv_3[t, lev_Zc[t]+2]               = qv[0, lev_Zc[t]-1] + ((self.Zc[t] - m_zc[lev_Zc[t]-1])
                                                                                / self.slope_qv[lev_Zc[t]-1])
                self.qv_3[t, lev_Zc[t]+3:]              = qv[0, lev_Zc[t]:]
                
                # ========== parcel ==========
                diff_z = np.diff(self.m_zc_3[t])
                self.temp_d_3[t,:] = thermo.th_2_T(self.th_m[t], self.p_zc_3[t, :])
                
                self.temp_parc[t, :lev_Zc[t]+3] = self.temp_d_3[t, :lev_Zc[t]+3]
                self.qvs_parc[t, :lev_Zc[t]+3]  = \
                    thermo.e_2_qv(thermo.cc_equation(self.temp_parc[t, :lev_Zc[t]+3]), 
                                    self.p_zc_3[t, :lev_Zc[t]+3])
                
                for lev in range(lev_Zc[t]+3, length_3 - 1):
                    self.temp_parc[t, lev], self.qvs_parc[t, lev] = moist_adiabatic(lev)
                        
                self.qv_parc[t, :lev_Zc[t]+3] = self.qvm[t]
                self.qv_parc[t, lev_Zc[t]+3:] = self.qvs_parc[t, lev_Zc[t]+3:]
                
                Tc = self.temp_3[t, lev_Zc[t]+2]
                
            else: # Zc < Zm
                # ========== env ==========
                self.m_zc_3[t, :lev_Zc[t]]              = m_zc[:lev_Zc[t]]
                self.m_zc_3[t, lev_Zc[t]]               = self.Zc[t]
                self.m_zc_3[t, lev_Zc[t]+1:lev_Zm[t]+1] = m_zc[lev_Zc[t]:lev_Zm[t]]
                self.m_zc_3[t, lev_Zm[t]+1:lev_Zm[t]+3] = self.Zm[t]
                self.m_zc_3[t, lev_Zm[t]+3:]            = m_zc[lev_Zm[t]:]
                
                self.p_zc_3[t, :lev_Zc[t]]              = p_zc[:lev_Zc[t]]
                self.p_zc_3[t, lev_Zc[t]]               = p_zc[lev_Zc[t]-1] + ((self.Zc[t] - m_zc[lev_Zc[t]-1])
                                                                                * self.slope_p[lev_Zc[t]-1])
                self.p_zc_3[t, lev_Zc[t]+1:lev_Zm[t]+1] = p_zc[lev_Zc[t]:lev_Zm[t]]
                self.p_zc_3[t, lev_Zm[t]+1:lev_Zm[t]+3] = p_zc[lev_Zm[t]-1] + ((self.Zm[t] - m_zc[lev_Zm[t]-1])
                                                                                * self.slope_p[lev_Zm[t]-1])
                self.p_zc_3[t, lev_Zm[t]+3:]            = p_zc[lev_Zm[t]:]
                
                self.th_3[t, :lev_Zm[t]+3]              = self.th_m[t]
                self.th_3[t, lev_Zm[t]+3:]              = th[0, lev_Zm[t]:]
                
                self.temp_3[t, :]                       = thermo.th_2_T(self.th_3[t, :], self.p_zc_3[t, :])
                self.qvs_3[t, :]                        = thermo.e_2_qv(thermo.cc_equation(self.temp_3[t, :]), 
                                                                         self.p_zc_3[t, :])
                
                self.qv_3[t, :lev_Zm[t]+2]              = self.qvm[t]
                self.qv_3[t, lev_Zm[t]+2]               = self.qv_Zm[t]
                self.qv_3[t, lev_Zm[t]+3:]              = qv[0, lev_Zm[t]:]
                
                # over saturated part
                for os_lev in range(lev_Zc[t]+1, lev_Zm[t]+2):
                    
                    def mid_point_method(lev):
                        tmp_qv_r = self.qvm[t]
                        tmp_qv_l = self.qvs_3[t, lev]
                        tmp_qvs  = (tmp_qv_l + tmp_qv_r) / 2
                        
                        # return error, tmp_temp
                        def F_check(tmp_qvs):
                            delta_qv = (self.qvm[t] - tmp_qvs)
                            # delta_qv = (tmp_qv_r - tmp_qvs) wrong
                            delta_T  = delta_qv * thermo.Lv / thermo.Cp
                            test_temp = self.temp_3[t, lev] + delta_T
                            test_qvs  = thermo.e_2_qv(thermo.cc_equation(test_temp), self.p_zc_3[t, lev])
                            
                            error = tmp_qvs - test_qvs
                            return error, test_temp

                        error, tmp_temp = F_check(tmp_qvs)
                        while abs(error) > 1e-6:

                            if error > 0:
                                tmp_qv_r = tmp_qvs
                                tmp_qvs  = (tmp_qv_l + tmp_qv_r) / 2
                                error, tmp_temp = F_check(tmp_qvs)
                                
                            else:
                                tmp_qv_l = tmp_qvs
                                tmp_qvs  = (tmp_qv_l + tmp_qv_r) / 2
                                error, tmp_temp = F_check(tmp_qvs)
                
                        return tmp_qvs, tmp_temp
                    
                    revise_qvs, revise_temp = mid_point_method(os_lev)
                    self.qv_3[t, os_lev]    = revise_qvs
                    self.qc_3[t, os_lev]    = self.qvm[t] - self.qv_3[t, os_lev]
                    self.temp_3[t, os_lev]  = revise_temp

                self.th_3[t, :] = thermo.T_2_th(self.temp_3[t, :], self.p_zc_3[t, :])
                Tc = self.temp_3[t, lev_Zc[t]]

                # ========== parcel ==========
                diff_z = np.diff(self.m_zc_3[t])
                self.temp_d_3[t,:] = thermo.th_2_T(self.th_m[t], self.p_zc_3[t, :])
                
                self.temp_parc[t, :lev_Zc[t]+1]   = self.temp_d_3[t, :lev_Zc[t]+1]
                self.qvs_parc[t, :lev_Zc[t]+1]    = \
                    thermo.e_2_qv(thermo.cc_equation(self.temp_parc[t, :lev_Zc[t]+1]), 
                                    self.p_zc_3[t, :lev_Zc[t]+1])
                
                for lev in range(lev_Zc[t]+1, length_3 - 1):
                    self.temp_parc[t, lev], self.qvs_parc[t, lev] = moist_adiabatic(lev)
                        
                self.qv_parc[t, :lev_Zc[t]+1] = self.qvm[t]
                self.qv_parc[t, lev_Zc[t]+1:] = self.qvs_parc[t, lev_Zc[t]+1:]
            
            self.qvs_3[t,:]  = thermo.e_2_qv(thermo.cc_equation(self.temp_3[t,:]), self.p_zc_3[t,:])
            self.th_e[t,:]   = thermo.theta_e_ac(self.th_3[t,:], Tc, self.qv_3[t,:])
            self.th_es[t,:]  = thermo.theta_e_ac(self.th_3[t,:], Tc, self.qvs_3[t,:])
            self.th_e_parc[t,:] = thermo.theta_e_ac(thermo.T_2_th(self.temp_parc[t,:], self.p_zc_3[t,:]), 
                                                     Tc, self.qv_parc[t,:])
            self.th_es_parc[t,:] = thermo.theta_e_ac(thermo.T_2_th(self.temp_parc[t,:], self.p_zc_3[t,:]), 
                                                     Tc, self.qvs_parc[t,:])
            
            pass # for i in range(145)
        
        # for CA8
        self.Tv_env   = np.zeros((145, length_3))
        self.Tv_parc  = np.zeros((145, length_3))
        self.hms_env  = np.zeros((145, length_3))
        self.hms_parc = np.zeros((145, length_3))
        for i in range(145):
            self.Tv_env[i,:]  = thermo.T_2_Tv(self.temp_3[i,:], self.qv_3[i,:])
            self.Tv_parc[i,:] = thermo.T_2_Tv(self.temp_parc[i,:], self.qv_parc[i,:])
            self.hms_env[i,:]  = self.temp_3[i,:]  + thermo.g * self.m_zc_3[i,:] / thermo.Cp\
                               + self.qv_3[i,:] * thermo.Lv / thermo.Cp
            self.hms_parc[i,:] = self.temp_parc[i,:]  + thermo.g * self.m_zc_3[i,:] / thermo.Cp\
                               + self.qv_parc[i,:] * thermo.Lv / thermo.Cp
            
        return None

# ==================================================
# main
def main():
    global locate, thermo, vvm, mix

    filepath = '..\\CA7\\files'
    locate = Location(filepath, name = 'Quark')
    # locate = Location(id = 'B12209017')
    thermo = Thermo()
    vvm    = VVM(locate.file)
    mix    = Mix()
       
    # for CA8
    # I hate CAPE and CIN
    def CAPE_CIN():
        
        def find_intersect(lev_Zm) -> list:
            reverse   = True # if Tv_env > Tv_parc
            tmp_inter_h  = []
            tmp_inter_Tv = []
            
            # I hate CAPE and CIN * 2
            lev = len(vvm.m_zc) -1 +3 # -1(Python) +3(insert 3)
            while True:
                if reverse:
                    while mix.Tv_env[t, lev] > mix.Tv_parc[t, lev]:
                        lev -= 1
                    # I hate CAPE and CIN * 3
                    if lev < lev_Zm:
                            return tmp_inter_h, tmp_inter_Tv
                    
                    slope_env  = (mix.Tv_env[t, lev+1] - mix.Tv_env[t, lev])\
                            / (mix.m_zc_3[t, lev+1] - mix.m_zc_3[t, lev])
                    slope_prac = (mix.Tv_parc[t, lev+1] - mix.Tv_parc[t, lev])\
                            / (mix.m_zc_3[t, lev+1] - mix.m_zc_3[t, lev])
                    
                    dz = (mix.Tv_parc[t, lev] - mix.Tv_env[t, lev])\
                    / (slope_env - slope_prac)
                    
                    tmp_inter_h.append(mix.m_zc_3[t, lev] + dz)
                    tmp_inter_Tv.append(mix.Tv_env[t, lev] + dz*slope_env)
                    reverse = False
                
                else:
                    while mix.Tv_env[t, lev] < mix.Tv_parc[t, lev]:
                        lev -= 1
                    
                    if lev < lev_Zm:
                        return tmp_inter_h, tmp_inter_Tv
                    
                    slope_env  = (mix.Tv_env[t, lev+1] - mix.Tv_env[t, lev])\
                            / (mix.m_zc_3[t, lev+1] - mix.m_zc_3[t, lev])
                    slope_prac = (mix.Tv_parc[t, lev+1] - mix.Tv_parc[t, lev])\
                            / (mix.m_zc_3[t, lev+1] - mix.m_zc_3[t, lev])
                    
                    dz = (mix.Tv_env[t, lev] - mix.Tv_parc[t, lev])\
                        / (slope_prac - slope_env)
                        
                    tmp_inter_h.append(mix.m_zc_3[t, lev] + dz)
                    tmp_inter_Tv.append(mix.Tv_env[t, lev] + dz*slope_env)
                    reverse = True
        
        def area_integral(t, lev, **kwargs) -> float:
            if 'upper' in kwargs:
                endpoint = kwargs['upper']
                area = (mix.Tv_parc[t, lev] - mix.Tv_env[t, lev]) / mix.Tv_env[t, lev]\
                     * (endpoint - mix.m_zc_3[t, lev]) / 2
                     
            elif 'lower' in kwargs:
                endpoint = kwargs['lower']
                area = (mix.Tv_parc[t, lev+1] - mix.Tv_env[t, lev+1]) / mix.Tv_env[t, lev+1]\
                     * (mix.m_zc_3[t, lev+1] - endpoint) / 2
                     
            else:
                area = ((mix.Tv_parc[t, lev+1] - mix.Tv_env[t, lev+1]) / mix.Tv_env[t, lev+1]\
                     +  (mix.Tv_parc[t, lev] - mix.Tv_env[t, lev]) / mix.Tv_env[t, lev]) \
                     * (mix.m_zc_3[t, lev+1] - mix.m_zc_3[t, lev]) / 2
                     
            return area * thermo.g
        
        EL           = np.zeros(145)
        LFC          = np.zeros(145)
        CAPE         = np.zeros(145)
        CIN          = np.zeros(145)
        intersect_h  = []
        intersect_Tv = []
        CAPE_range   = []
        CIN_range    = []
        
        for t in range(145):
            lev_Zm = np.searchsorted(mix.m_zc_3[t], mix.Zm[t], side='right')
            intersect_h.append(find_intersect(lev_Zm)[0])
            intersect_Tv.append(find_intersect(lev_Zm)[1])
            
            if intersect_h[t][-1] < mix.Zc[t]:
                LFC[t] = mix.Zc[t]
            elif len(intersect_h[t]) == 1:
                LFC[t] = mix.Zm[t]
            else:
                LFC[t] = intersect_h[t][-1]
            
            EL[t] = intersect_h[t][0]
            
            # ========== CAPE ==========
            idx = np.searchsorted(mix.m_zc_3[t], intersect_h[t][0], side='right') - 1
            tmp_range = []
            tmp_range.append(EL[t])
            CAPE[t] += area_integral(t, idx, upper=intersect_h[t][0])
            
            while mix.m_zc_3[t][idx] > LFC[t]:
                tmp_range.append(mix.m_zc_3[t][idx])
                idx -= 1
                CAPE[t] += area_integral(t, idx)
            
            idx -= 1
            CAPE[t] += area_integral(t, idx, lower=mix.Zm[t])
            tmp_range.append(LFC[t])
            
            # ========== CIN ==========
            idx = np.searchsorted(mix.m_zc_3[t], LFC[t], side='right') - 1
            tmp_range = []
            tmp_range.append(EL[t])
            tmp = area_integral(t, idx, upper=LFC[t])
            if tmp < 0:
                    CIN[t] += tmp
                    
            while mix.m_zc_3[t][idx] > vvm.m_zc[0]:
                tmp_range.append(mix.m_zc_3[t][idx])
                idx -= 1
                tmp = area_integral(t, idx)
                if tmp < 0:
                    CIN[t] += tmp
            
            idx -= 1
            tmp = area_integral(t, idx, lower=mix.Zm[t])
            if tmp < 0:
                    CIN[t] += tmp
            tmp_range.append(vvm.m_zc[0])

            CAPE_range.append(tmp_range)
            CIN_range.append(tmp_range)
                
        return EL, LFC, CAPE_range, CIN_range, intersect_h, intersect_Tv, CAPE, CIN
    
    global EL, LFC, CAPE_range, CIN_range, intersect_h, intersect_Tv, CAPE, CIN
    EL, LFC, CAPE_range, CIN_range, intersect_h, intersect_Tv, CAPE, CIN\
        = CAPE_CIN()
    
    def test():
        def AAAAA(t):
            T   = mix.temp_3[t,:] * units.K
            Tp  = mix.temp_parc[t,:] * units.K
            p   = mix.p_zc_3[t,:] * units.hPa
            rh  = (mix.qv_3[t,:] / mix.qvs_3[t,:] * 100) * units.percent
            qv  = mix.qv_3[t,:] * units('kg/kg')
            
            # Td  = dewpoint()
            # Td = dewpoint_from_specific_humidity(p, T, qv)
            Td  = dewpoint_from_relative_humidity(T, rh)
            cape, cin = cape_cin(p, T, Td, Tp)
            
            return cape.magnitude, cin.magnitude
        
        cape = [AAAAA(t)[0] for t in range(145)]
        cin  = [AAAAA(t)[1] for t in range(145)]
        
        fig,ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(cape)
        ax[0].plot(CAPE)
        ax[1].plot(cin)
        ax[1].plot(CIN)
        
        plt.show()
    
    def COIN():
        f,ax = plt.subplots(figsize=(6, 3))
        f.subplots_adjust(top=0.8, bottom=0.3)
        f.suptitle('time series of COIN', y = 0.9)

        ax.plot(np.linspace(0, 24, 145), CAPE/(CAPE-CIN), '-b', label='COIN')
        ax.set_xlabel('Time [LST]')
        ax.set_xlim(0, 24)
        ax.set_xticks(np.linspace(0, 24, 7))
        # ax.set_ylim(0, 1)
        ax.grid()
        ax.legend(loc = 'lower right')
        f.savefig('CA8_CAPE.png', dpi=300)
    
    test()
    # Draw('CA8', frames = 144, save = True)
    return None

# ==================================================
class Draw:
    
    def __init__(self, dtype: str, frames: int = 24, save: bool = False, **kwargs) -> None:
        self.t_step = np.linspace(0, 144, 145)
        self.frames = frames
            
        if dtype == 'CA8':
            anim = self.CA8_animation(self.frames)
        elif dtype == 'SkewT':
            anim = self.SkewT(self.frames)
                
        if save:
            if 'name' in kwargs:
                self.name = kwargs['name']
            else:
                self.name = dtype
            self.saveanim(anim, self.frames)
        else:
            plt.show()
            
    def saveanim(self, anim: FuncAnimation, frames: int):
        FFwriter = FFMpegWriter(fps = frames/12)
        anim.save(f'videos/0{self.name}_{locate.file_name}_{frames}f.mp4', 
                  writer=FFwriter, dpi=200)

    def CA8_animation(self, frames):
        # ========== fig setup ==========
        def setup():
            fig = plt.figure(figsize=(9, 6))
            fig.subplots_adjust(top=0.8)
            gs = GridSpec(3, 4, figure=fig)
            
            ax = {'Tv'  : fig.add_subplot(gs[:, :2]),
                  'CAPE': fig.add_subplot(gs[0, 2:]),
                  'CIN' : fig.add_subplot(gs[1, 2:]),
                  'K'   : fig.add_subplot(gs[2, 2]),
                  'qv'  : fig.add_subplot(gs[2, 3]),}

            axpos = [axes.get_position() for axes in ax.values()]
            
            ax['Tv'].set_position([axpos[0].x0, axpos[0].y0, 
                                  axpos[0].width*0.85, axpos[0].height])
            ax['CAPE'].tick_params(bottom=False, labelbottom=False)
            ax['CAPE'].set_position([axpos[1].x0, axpos[1].y0 + (axpos[1].height * 0.2), 
                                     axpos[1].width, axpos[1].height * 0.8])
            ax['CIN'] .set_position([axpos[2].x0, axpos[2].y0 + (axpos[1].height * 0.5), 
                                     axpos[2].width, axpos[2].height * 0.8])
            ax['CIN'] .sharex(ax['CAPE'])
            
            ax['K'] .set_position([axpos[3].x0, axpos[3].y0, 
                                   axpos[3].width, axpos[3].height * 1.1])
            ax['K'] .sharey(ax['qv'])
            ax['qv'].set_position([axpos[4].x0, axpos[4].y0, 
                                   axpos[4].width, axpos[4].height * 1.1])
            ax['qv'].tick_params(left=False, labelleft=False)
            
            return fig, ax
            
        fig, ax = setup()
        tmp = np.max(np.hstack((mix.Zc, mix.Zm))) * 1.333
        low_lim = np.searchsorted(vvm.m_zc, tmp, side='right')
        top_lim = -3
        # ========== lines init ==========
        # Tv
        L_env_Tv, = ax['Tv'].plot([], [], '-b', lw=1, label='env. Tv')
        L_parc_Tv, = ax['Tv'].plot([], [], '--r', lw=1, label='parcel Tv')
        
        legend_cape = ax['CAPE'].fill_betweenx([], [], color='r', alpha=0.2, 
                                               interpolate=True, label='CAPE')
        legend_cin  = ax['CIN'].fill_betweenx([], [], color='b', alpha=0.2, 
                                              interpolate=True , label='CIN')
        
        # CAPE and CIN
        L_CAPE, = ax['CAPE'].plot(self.t_step, CAPE, '-r', lw=1)
        L_CIN,  = ax['CIN'].plot(self.t_step, -CIN, '-b', lw=1)

        # Temp.
        L_env_temp, = ax['K'].plot([], [], '-b', lw=1)
        L_parc_temp, = ax['K'].plot([], [], '--r', lw=1)
        
        # qv
        L_env_qv, = ax['qv'].plot([], [], '-b', lw=1)
        L_parc_qv, = ax['qv'].plot([], [], '--r', lw=1)
        
        # ========== more setting :( ==========
        # Tv
        ax['Tv'].set_title('Tv', loc = 'left')
        ax['Tv'].set_xlabel(r'[K]')
        ax['Tv'].set_ylabel(r'Height [m]')
        ax['Tv'].set_ylim(vvm.m_zc[0], vvm.m_zc[top_lim-1])
        ax['Tv'].set_yticks(Q_draw.autoticks(vvm.m_zc[0], vvm.m_zc[top_lim-1], 100, 8, side='left'))
        ax['Tv'].legend(handles=[L_env_Tv, L_parc_Tv], 
                        bbox_to_anchor=(1, 1), loc='lower right', ncol=2)
        
        tmp_xlim = (np.min(mix.Tv_env[:, :top_lim]) - 5, np.max(mix.Tv_env[:, :top_lim]) + 2)
        ax['Tv'].set_xlim(tmp_xlim)
        
        L_EL   = ax['Tv'].axhline(0, c='k', lw=1, ls='--')
        txt_EL = ax['Tv'].text(0, 0, '', fontsize=10, ha='right')

        # CAPE / CIN
        ax['CAPE'].set_title(r'[ $J \cdot kg^{-1}$]', loc = 'left')
        ax['CIN'].set_xlabel('Time [LST]')
        ax['CIN'].set_xticks(self.t_step[::24], labels=(self.t_step[::24]/6).astype(int))
        ax['CIN'].set_xlim(0, 144)
        ax['CAPE'].legend(handles=[L_CAPE, L_CIN], labels=['CAPE', 'CIN'],
                          bbox_to_anchor=(1, 1), loc='lower right', ncol=2)
        
        # K / qv
        ax['K'].set_title(r'Temp.', loc = 'left')
        ax['K'].set_xlabel(r'[K]')
        ax['qv'].set_title(r'qv', loc = 'left')
        ax['qv'].set_xlabel(r'[g/kg]')
        ax['K'].set_ylabel(r'Height [m]')
        
        tmp_xlim = (np.min(mix.temp_3[:, :low_lim+3]), np.max(mix.temp_3[:, :low_lim+3]))
        ax['K'].set_xlim(tmp_xlim)
        tmp_x = np.min(mix.qv_3[:, :low_lim+3])
        tmp_xlim = (tmp_x, tmp_x + (np.max(mix.qvm) - tmp_x)*1.5)
        ax['qv'].set_xlim(tmp_xlim)
        
        txt_Zm = ax['qv'].text(0, 0, '', fontsize=8, ha='right')
        txt_Zc = ax['K'].text(0, 0, '', fontsize=8, ha='right')
        
        lines_zc = []
        lines_zm = []
        for key in ['K', 'qv']:
            ax[key].set_ylim(vvm.m_zc[0], vvm.m_zc[low_lim-1])
            ax[key].set_yticks(Q_draw.autoticks(vvm.m_zc[0], vvm.m_zc[low_lim-1], 50, 6, side='left'))
            lines_zc.append(ax[key].axhline(mix.Zc[0], c='k', lw=1, ls='--'))
            lines_zm.append(ax[key].axhline(mix.Zm[0], c='k', lw=1, ls='--'))
      
        ax['axins'] = ax['Tv'].inset_axes(
            [0.01, 0.01, .45, .35],
            xlim=(np.min(mix.Tv_parc[:, :low_lim+13]), np.max(mix.Tv_parc[:, :low_lim+13])), 
            ylim=(vvm.m_zc[0], vvm.m_zc[low_lim+9]))
        ax['axins'].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False,
                                top=True, labeltop=True, right=True, labelright=True,
                                labelsize=8)
        
        txt_LFC = ax['axins'].text(0, 0, '', fontsize=8, ha='right')
        
        l_env_Tv, = ax['axins'].plot([], [], '-b', lw=1)
        l_parc_Tv, = ax['axins'].plot([], [], '--r', lw=1)
        l_LFC = ax['axins'].axhline(0, c='k', lw=1, ls='--')
    
        for axes in ax.values():
            axes.grid()
        # ==================================================
        # animating
        def update(t):
            for artist in ax['Tv'].collections + ax['axins'].collections:
                artist.remove()
            
            # # I hate CAPE and CIN
            # # shade CAPE
            # tmp_xl, tmp_xr = [intersect_Tv[t][0]], [intersect_Tv[t][0]]
            
            # for h in CAPE_range[t][1:-1]:
            #     idx = np.argmin(np.abs(mix.m_zc_3[t] - h))
            #     tmp_xl.append(mix.Tv_env[t, idx])
            #     tmp_xr.append(mix.Tv_parc[t, idx])
            
            # i = np.argmin(np.abs(intersect_h[t] - LFC[t]))
            # tmp_xl.append(intersect_Tv[t][i])
            # tmp_xr.append(intersect_Tv[t][i])
            
            # ax['Tv'].fill_betweenx(CAPE_range[t], tmp_xl, tmp_xr, color='r', alpha=0.2)
            # ax['axins'].fill_betweenx(CAPE_range[t], tmp_xl, tmp_xr, color='r', alpha=0.2)
            
            # # shade CIN
            # tmp_xl, tmp_xr = [intersect_Tv[t][i]], [intersect_Tv[t][i]]
            
            # for h in CIN_range[t][1:]:
            #     idx = np.argmin(np.abs(mix.m_zc_3[t] - h))
            #     tmp_xl.append(mix.Tv_parc[t, idx])
            #     tmp_xr.append(mix.Tv_env[t, idx])
            
            # ax['Tv'].fill_betweenx(CIN_range[t], tmp_xr, tmp_xl, color='b', alpha=0.2)
            # ax['axins'].fill_betweenx(CIN_range[t], tmp_xr, tmp_xl, color='b', alpha=0.2)
            
            '''for range_ in CAPE_range[t]:
                tmp_xl, tmp_xr = [intersect_Tv[t][i]], [intersect_Tv[t][i]]
                            
                for h in range_[1:-1]:
                    idx = np.argmin(np.abs(mix.m_zc_3[t] - h))
                    tmp_xl.append(mix.Tv_env[t, idx])
                    tmp_xr.append(mix.Tv_parc[t, idx])
                    
                if i == len(intersect_h[t]) - 1:
                    idx = np.argmin(np.abs(mix.m_zc_3[t] - mix.Zm[t]))
                    tmp_xl.append(mix.Tv_env[t, idx+1])
                    tmp_xr.append(mix.Tv_parc[t, idx+1])
                else:
                    tmp_xl.append(intersect_Tv[t][i+1])
                    tmp_xr.append(intersect_Tv[t][i+1])
        
                ax['Tv'].fill_betweenx(range_, tmp_xl, tmp_xr, color='r', alpha=0.2)
                ax['axins'].fill_betweenx(range_, tmp_xl, tmp_xr, color='r', alpha=0.2)
                i += 2
    
            # shade CIN
            for range_ in CIN_range[t]:
                i = len(intersect_h[t]) - 1
                tmp_xl, tmp_xr = [intersect_Tv[t][i]], [intersect_Tv[t][i]]
                
                for h in range_[1:-1]:
                    idx = np.argmin(np.abs(mix.m_zc_3[t] - h))
                    tmp_xl.append(mix.Tv_parc[t, idx])
                    tmp_xr.append(mix.Tv_env[t, idx])
                    
                if i == len(intersect_h[t]) - 1:
                    idx = np.argmin(np.abs(mix.m_zc_3[t] - mix.Zm[t]))
                    tmp_xl.append(mix.Tv_parc[t, idx+1])
                    tmp_xr.append(mix.Tv_env[t, idx+1])
                else:
                    tmp_xl.append(intersect_Tv[t][i+1])
                    tmp_xr.append(intersect_Tv[t][i+1])
                
                ax['Tv'].fill_betweenx(range_, tmp_xl, tmp_xr, color='b', alpha=0.2)
                ax['axins'].fill_betweenx(range_, tmp_xl, tmp_xr, color='b', alpha=0.2)'''
                                    
            L_env_Tv.set_data(mix.Tv_env[t, :top_lim], mix.m_zc_3[t, :top_lim])
            L_parc_Tv.set_data(mix.Tv_parc[t, :top_lim], mix.m_zc_3[t, :top_lim])
            l_env_Tv.set_data(mix.Tv_env[t, :low_lim+13], mix.m_zc_3[t, :low_lim+13])
            l_parc_Tv.set_data(mix.Tv_parc[t, :low_lim+13], mix.m_zc_3[t, :low_lim+13])
            l_LFC.set_ydata(LFC[t])
            
            L_EL.set_ydata(EL[t])
            txt_EL.set_position((ax['Tv'].get_xlim()[1], EL[t]+50))
            txt_EL.set_text('EL:{:.1f}m'.format(EL[t]))
            txt_LFC.set_position((ax['axins'].get_xlim()[1], LFC[t]+50))
            txt_LFC.set_text('LFC:{:.1f}m'.format(LFC[t]))
            
            L_env_temp.set_data(mix.temp_3[t, :low_lim+3], mix.m_zc_3[t, :low_lim+3])
            L_parc_temp.set_data(mix.temp_parc[t, :low_lim+3], mix.m_zc_3[t, :low_lim+3])
            
            L_env_qv.set_data(mix.qv_3[t, :low_lim+3], mix.m_zc_3[t, :low_lim+3])
            L_parc_qv.set_data(mix.qv_parc[t, :low_lim+3], mix.m_zc_3[t, :low_lim+3])
            
            txt_Zm.set_position((ax['qv'].get_xlim()[1], mix.Zm[t]+5))
            txt_Zm.set_text('Zm:{:.1f}m'.format(mix.Zm[t]))
            txt_Zc.set_position((ax['K'].get_xlim()[1], mix.Zc[t]+5))
            txt_Zc.set_text('Zc:{:.1f}m'.format(mix.Zc[t]))
            
            for line in lines_zc:
                line.set_ydata(mix.Zc[t])
            for line in lines_zm:
                line.set_ydata(mix.Zm[t])
            
            fig.suptitle('TaiwanVVM simulation, tpe20110802cln\nTime: %02d:%1d0 LST @(%8sE, %8sN)' 
                          %(int(t/6), np.mod(t, 6), locate.loc[0], locate.loc[1]), 
                          x=0.1, y=0.975, ha = 'left')
            
            print('\rFrame:{}'.format(t), end='')
            return None
        
        return (FuncAnimation(fig, update, repeat=True,
           frames=range(0, 145, int(144/frames)), interval=10000/frames))

# ==================================================
# execute
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))
# ==================================================
# Last CA, finally ...( _ _)zzz