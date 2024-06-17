# CA7 / mixed layer
# ==================================================
import os
import sys
import time
import netCDF4              as nc
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gs
import sympy                as sym
from sympy.solvers          import nsolve
from sympy                  import Symbol
from scipy                  import integrate
from matplotlib.animation   import *

sys.path.append('..')
from Q_package.CA_ import Thermo, Location
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
                    tmp_gamma = thermo.gamma_m(self.temp_parc[t, lev-1], self.qvs_parc[t, lev-1])
                    
                    self.temp_parc[t, lev] = self.temp_parc[t, lev-1] - (diff_z[lev-1] * tmp_gamma)
                    self.qvs_parc[t, lev] = \
                        thermo.e_2_qv(thermo.cc_equation(self.temp_parc[t, lev]), 
                                        self.p_zc_3[t, lev])
                        
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
                    tmp_gamma = thermo.gamma_m(self.temp_parc[t, lev-1], self.qvs_parc[t, lev-1])
                    self.temp_parc[t, lev] = self.temp_parc[t, lev-1] - (diff_z[lev-1] * tmp_gamma)
                    self.qvs_parc[t, lev] = \
                        thermo.e_2_qv(thermo.cc_equation(self.temp_parc[t, lev]), 
                                        self.p_zc_3[t, lev])
                        
                self.qv_parc[t, :lev_Zc[t]+1] = self.qvm[t]
                self.qv_parc[t, lev_Zc[t]+1:] = self.qvs_parc[t, lev_Zc[t]+1:]
            
            self.th_es_parc[t,:] = thermo.theta_e_ac(thermo.T_2_th(self.temp_parc[t,:], self.p_zc_3[t,:]), 
                                                     Tc, self.qvs_parc[t,:])
            self.qvs_3[t,:]  = thermo.e_2_qv(thermo.cc_equation(self.temp_3[t,:]), self.p_zc_3[t,:])
            self.th_e[t,:]   = thermo.theta_e_ac(self.th_3[t,:], Tc, self.qv_3[t,:])
            self.th_es[t,:]  = thermo.theta_e_ac(self.th_3[t,:], Tc, self.qvs_3[t,:])
            pass # for i in range(145)
    
        return None

# ==================================================
# main   
def main():
    global locate, thermo, vvm, mix
    
    locate = Location(name = 'Quark')
    # locate = Location(id = 'B12209017')
    thermo = Thermo()
    vvm    = VVM(locate.file)
    mix    = Mix()
    
    def test(*args):
        for arr in args:
            plt.plot(arr)
        plt.show()
        return None

    # test(mix.Zc)
    # exit()
    # Draw('Est', save=True, frames=24)
    return None
    
# ==================================================
# Draw
class Draw:
    
    def __init__(self, dtype: str, frames: int = 24, save: bool = False, **kwargs) -> None:
        self.t_step = np.linspace(0, 144, 145)
        self.frames = frames
            
        if dtype == 'Est':
            anim = self.animate_Est(self.frames)
        elif dtype == 'VVM':
            anim = self.animate_VVM(self.frames)
                
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
        anim.save(f'videos/{self.name}_{locate.file_name}_{frames}f.mp4', 
                  writer=FFwriter, dpi=200)


    def animate_Est(self, frames):
        
        fig,ax = plt.subplots(2, 2, figsize = (8,6), sharey=True)
        fig.subplots_adjust(hspace=0.5, top=0.8, right=0.9, left=0.15)
        tmp = np.max(np.hstack((mix.Zc, mix.Zm))) * 1.333
        top_lim = np.searchsorted(vvm.m_zc, tmp, side='right') + 0
        
        ''' not used
        ax1pos = ax[0,0].get_position()
        ax2pos = ax[0,1].get_position()
        ax3pos = ax[1,0].get_position()
        ax4pos = ax[1,1].get_position()
        ax[0,0].set_position([ax1pos.x0, ax1pos.y0, ax1pos.width*1.075, ax1pos.height])
        ax[0,1].set_position([ax2pos.x0-(ax2pos.width*1.075-ax2pos.width), ax2pos.y0, ax2pos.width*1.075, ax2pos.height])
        ax[1,0].set_position([ax3pos.x0, ax3pos.y0, ax3pos.width*0.75, ax3pos.height])
        ax[1,1].set_position([ax4pos.x0-.1, ax4pos.y0, ax4pos.width*0.75, ax4pos.height])
        
        ax[0,0].spines['right'].set_visible(False)
        ax[0,1].spines['left'].set_visible(False)
        ax[0,1].tick_params(axis='y', left=False)
        
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(d, -1), (-d, 1)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax[0,0].plot([1, 1], [0, 1], transform=ax[0,0].transAxes, **kwargs)
        ax[0,1].plot([0, 0], [0, 1], transform=ax[0,1].transAxes, **kwargs)
        '''
        
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
            
        ax[0,0].set_ylim([vvm.m_zc[0], vvm.m_zc[top_lim-1]])
        ax[0,0].set_yticks(autoticks(vvm.m_zc[0], vvm.m_zc[top_lim-1], 100, 6))
        fig.text(0.04, 0.5, 'Height [m]', fontsize=12, va='center', rotation='vertical')
        
        # Temp and theta
        L_mix_th,    = ax[0,0].plot(mix.th_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-r', lw=1)
        L_bef_th,    = ax[0,0].plot(mix.th_2[0, :top_lim+2], mix.m_zc_2[0, :top_lim+2], ':r', lw=1)
        L_mix_Temp,  = ax[0,0].plot(mix.temp_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-b', lw=1)
        L_bef_Temp,  = ax[0,0].plot(mix.temp_2[0, :top_lim+2], mix.m_zc_2[0, :top_lim+2], ':b', lw=1)
        L_parc_Temp, = ax[0,0].plot(mix.temp_parc[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '--m', lw=1)
        
        ax[0,0].set_title(r'$Temp.$ / $\theta$ / $\theta_{e}$ / $\theta_{es}$', loc='left')
        xlim_tmp = (np.min(mix.temp_3[0, :top_lim+3])-1, np.max(mix.th_3[0, :top_lim+3])+2)
        ax[0,0].set_xticks(autoticks(xlim_tmp[0], xlim_tmp[1], 1, 6, side='left'))
        ax[0,0].set_xlim(xlim_tmp)
        ax[0,0].set_xlabel(r'$[K]$')
        
        # theta_e and theta_es
        L_mix_th_e,  = ax[0,1].plot(mix.th_e[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-b', lw=1, label=r'Est. $\theta_e$')
        L_mix_th_es, = ax[0,1].plot(mix.th_es[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-g', lw=1, label=r'Est. $\theta_{es}$')
        
        L_parc_th_es, = ax[0,1].plot(mix.th_es_parc[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '--m', lw=1, label=r'Parc. $\theta_{es}$')
        
        xlim_tmp = (np.min(mix.th_e[:, :top_lim+3])-1, np.max(mix.th_es[:, :top_lim+3])+1)
        ax[0,1].set_xticks(autoticks(xlim_tmp[0], xlim_tmp[1], 1, 6, side='left'))
        ax[0,1].set_xlim(xlim_tmp)
        ax[0,1].set_xlabel(r'$[K]$')
        ax[0,1].legend(handles = [L_mix_th, L_bef_th, L_mix_Temp, L_bef_Temp, L_mix_th_e, L_mix_th_es],
                       labels=[r'$\theta_{ after}$', r'$\theta_{ before}$', r'$Temp._{ after}$', r'$Temp._{ before}$', r'$\theta_e$', r'$\theta_{es}$'],
                       ncol=3, loc='lower right', bbox_to_anchor=(1, 1), fontsize=8)
        
        # qv and qt
        ax[1,0].plot(vvm.qv[0, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='cyan', label='Init. qv')
        L_mix_qt,    = ax[1,0].plot(mix.qv_2[0, :top_lim+2], mix.m_zc_2[0, :top_lim+2], '--m', lw=1, label='Est. qt')
        L_mix_qv,    = ax[1,0].plot(mix.qv_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-b', lw=1, label='Est. qv')
        
        ax[1,0].set_title('qv / qt', loc='left')
        ax[1,0].set_xlim(np.min(mix.qv_3[:, :top_lim+3]), np.max(mix.qv_3[:, :top_lim+3])+.001)
        ax[1,0].set_xticklabels([f'{x:.1f}' for x in ax[1,0].get_xticks()*1000])
        ax[1,0].set_xlabel(r'[g/kg]')
        
        # qc
        ax[1,1].plot([0, 0], ax[1,1].get_ylim(), '-', lw=1, c='gray', label=None)
        L_mix_qc,    = ax[1,1].plot(mix.qc_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-b', lw=1, label='Est. qc')
        
        ax[1,1].set_title('ql', loc='left')
        ax[1,1].set_xlim(-0.0001, 0.001)
        ax[1,1].set_xticklabels([f'{x:.1f}' for x in ax[1,1].get_xticks()*1000])
        ax[1,1].set_xlabel(r'[g/kg]')
        ax[1,1].legend(handles = [L_mix_qc], labels=[r'$q_c$'], loc='upper right', fontsize=8)
        
        Zm_collect = []
        Zc_collect = []
        twin_ax = []
        for axes in ax.flatten():
            xlim = axes.get_xlim()
            # twin_ax.append(axes.twinx())
            Zm_collect.append(axes.plot(xlim, [mix.Zm[0], mix.Zm[0]], '--', lw=1, c='peru', zorder=0))
            Zc_collect.append(axes.plot(xlim, [mix.Zc[0], mix.Zc[0]], '--', lw=1, c='red', zorder=0))

            axes.grid()

        for axes in twin_ax:
            axes.set_ylim([vvm.m_zc[0], vvm.m_zc[top_lim-1]])
            axes.set_yticks([])     
        
        # update animation
        def update(t):
            
            for Line in Zm_collect:
                Line[0].set_ydata([mix.Zm[t], mix.Zm[t]])
            for Line in Zc_collect:
                Line[0].set_ydata([mix.Zc[t], mix.Zc[t]])
            
            for axes in [ax[1,0], ax[1,1]]:
                for artist in axes.collections:
                    artist . remove()
                
            ax[1,0].fill_betweenx(mix.m_zc_3[t, :top_lim+3], 
                                  mix.qv_3[t, :top_lim+3], mix.qv_3[t, :top_lim+3]+mix.qc_3[t, :top_lim+3], 
                                  color='blue', alpha=0.1)
            ax[1,1].fill_betweenx(mix.m_zc_3[t, :top_lim+3], 
                                  0, mix.qc_3[t, :top_lim+3], 
                                  color='blue', alpha=0.1)
            
            L_mix_th.set_data(mix.th_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
            L_bef_th.set_data(mix.th_2[t, :top_lim+2], mix.m_zc_2[t, :top_lim+2])
            L_mix_Temp.set_data(mix.temp_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
            L_bef_Temp.set_data(mix.temp_2[t, :top_lim+2], mix.m_zc_2[t, :top_lim+2])
            L_parc_Temp.set_data(mix.temp_parc[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
            
            L_mix_th_e.set_data(mix.th_e[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
            L_mix_th_es.set_data(mix.th_es[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
            
            L_parc_th_es.set_data(mix.th_es_parc[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
            
            L_mix_qt.set_data(mix.qv_2[t, :top_lim+2], mix.m_zc_2[t, :top_lim+2])
            L_mix_qv.set_data(mix.qv_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
                        
            L_mix_qc.set_data(mix.qc_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
            
            if fig.legends:
                for legend in fig.legends:
                    legend.remove()
                    
            fig.legend(handles=[Zm_collect[0][0], Zc_collect[0][0]], 
                       labels=[r'$Z_m$: ' + f'{mix.Zm[t]:.1f}m', r'$Z_c$: ' + f'{mix.Zc[t]:.1f}m'], 
                       loc='upper left', bbox_to_anchor=(0.725, 0.98), fontsize=10)
            
            fig.suptitle('TaiwanVVM simulation, tpe20110802cln\nTime: %02d:%1d0 LST @(%8sE, %8sN)' 
                          %(int(t/6), np.mod(t, 6), locate.loc[0], locate.loc[1]), 
                          x=0.02, y=0.98, ha = 'left')
            
            print('\rFrame:{}'.format(t), end='')
            return None
    
        return (FuncAnimation(fig, update, repeat=True,
           frames=range(0, 145, int(144/frames)), interval=10000/frames))
        
    def animate_VVM(self, frames):
        
        fig,ax  = plt.subplots(2, 4, figsize = (9, 6), sharey=True)
        twin_ax = ax[0,3].twinx()
        fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.8, right=0.85)
        tmp     = np.max(np.hstack((mix.Zc, mix.Zm))) * 1.333
        top_lim = np.searchsorted(vvm.m_zc, tmp, side='right')
        
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(d, -1), (-d, 1)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        
        d = 1.125
        for axes in ax.flatten()[::2]:
            axes_pos = axes.get_position()
            axes.set_position([axes_pos.x0, axes_pos.y0, axes_pos.width*d, axes_pos.height])
            axes.plot([1, 1], [0, 1], transform=axes.transAxes, **kwargs)
            axes.spines['right'].set_visible(False)
        
        for axes in ax.flatten()[1::2]:
            axes_pos = axes.get_position()
            axes.set_position([axes_pos.x0-(axes_pos.width*d-axes_pos.width), axes_pos.y0, axes_pos.width*d, axes_pos.height])
            axes.plot([0, 0], [0, 1], transform=axes.transAxes, **kwargs)
            axes.spines['left'].set_visible(False)
            axes.tick_params(axis='y', left=False)
        
        def ytick_(y_min, y_max):
            step = 0
            tmp  = 10
            while tmp > 6:
                step += 100
                array = np.arange(y_min, y_max, step)
                tmp   = len(array)
            return array
        
        # Temp and theta
        L_mix_th,    = ax[0,0].plot(mix.th_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-r', lw=1, label=r'Est. $\theta$')
        L_mix_Temp,  = ax[0,0].plot(mix.temp_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-b', lw=1, label='Est. Temp')
        L_th,    = ax[1,0].plot(vvm.th[0, :top_lim], vvm.m_zc[:top_lim], '-r', lw=1, label=None)
        L_Temp,  = ax[1,0].plot(vvm.temp[0, :top_lim],  vvm.m_zc[:top_lim], '-b', lw=1, label=None)
        
        # theta_e and theta_es
        L_mix_th_e,  = ax[0,1].plot(mix.th_e[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-', c='orange', lw=1, label=r'Est. $\theta_e$')
        L_mix_th_es, = ax[0,1].plot(mix.th_es[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-', c='green', lw=1, label=r'Est. $\theta_{es}$')
        L_th_e,  = ax[1,1].plot(vvm.th_e[0, :top_lim], vvm.m_zc[:top_lim], '-', c='orange', lw=1, label=r'VVM $\theta_e$')
        L_th_es, = ax[1,1].plot(vvm.th_es[0, :top_lim], vvm.m_zc[:top_lim], '-', c='green', lw=1, label=r'VVM $\theta_{es}$')

        # qv and qt
        L_mix_qt,    = ax[0,3].plot(mix.qv_2[0, :top_lim+2], mix.m_zc_2[0, :top_lim+2], '--m', lw=1, label='Est. qt')
        L_mix_qv,    = ax[0,3].plot(mix.qv_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-b', lw=1, label='Est. qv')
        L_qt,    = ax[1,3].plot(vvm.qv[0, :top_lim] + vvm.qc[0, :top_lim], vvm.m_zc[:top_lim], '--m', lw=1, label='VVM qvs')
        L_qv,    = ax[1,3].plot(vvm.qv[0, :top_lim], vvm.m_zc[:top_lim], '-b', lw=1, label='VVM qv')

        # qc
        L_mix_qc,   = ax[0,2].plot(mix.qc_3[0, :top_lim+3], mix.m_zc_3[0, :top_lim+3], '-g', lw=1, label='Est. qc')
        L_qc,    = ax[1,2].plot(vvm.qc[0, :top_lim], vvm.m_zc[:top_lim], '-g', lw=1, label='VVM qc')
        
        ax[0,0].set_ylim([vvm.m_zc[0], vvm.m_zc[top_lim-1]])
        ax[0,0].set_yticks(ytick_(vvm.m_zc[0], vvm.m_zc[top_lim-1]))
        ax[0,0].set_ylabel('Height [m]')
        ax[1,0].set_ylabel('Height [m]')
        
        ax[0,0].set_title(r'Est.', loc='left')
        ax[1,0].set_title(r'VVM', loc='left')
        
        tmp_xlim = [np.min(np.hstack((vvm.temp[:, :top_lim], mix.temp_3[:, :top_lim+3]))), 
                    np.max(np.hstack((vvm.th[:, :top_lim], mix.th_3[:, :top_lim+3])))+1]
        
        ax[0,0].set_xlim(tmp_xlim)
        ax[1,0].set_xlim(tmp_xlim)
        
        tmp_xlim = [np.min(np.hstack((vvm.th_e[:, :top_lim], mix.th_e[:, :top_lim+3]))), 
                    np.max(np.hstack((vvm.th_es[:, :top_lim], mix.th_es[:, :top_lim+3])))]
        
        ax[0,1].set_xlim(tmp_xlim)
        ax[1,1].set_xlim(tmp_xlim)
        ax[1,0].set_xlabel(r'$[K]$', loc='left')
        
        ax[0,1].legend(handles = [L_mix_th, L_mix_Temp, L_mix_th_e, L_mix_th_es],
                       labels=[r'$\theta$',  r'$Temp.r}$', r'$\theta_e$', r'$\theta_{es}$'],
                       ncol=2, loc='lower right', bbox_to_anchor=(1, 1), fontsize=8)
        
        ax[0,2].set_title(r'Est.', loc='left')
        ax[1,2].set_title(r'VVM', loc='left')
        
        tmp_xlim = [-0.0001, np.max(np.hstack((vvm.qc[:, :top_lim], mix.qc_3[:, :top_lim+3])))+0.0001]
        ax[0,2].set_xlim(tmp_xlim)
        ax[1,2].set_xlim(tmp_xlim)
        tmp_xlim = [np.min(np.hstack((vvm.qv[:, :top_lim], mix.qv_3[:, :top_lim+3]))), 
                    np.max(np.hstack((vvm.qv[:, :top_lim], mix.qv_3[:, :top_lim+3])))+.001]
        ax[0,3].set_xlim(tmp_xlim)
        ax[1,3].set_xlim(tmp_xlim)
        ax[1,2].set_xlabel(r'[g/kg]', loc='left')
        
        ax[0,2].set_xticklabels(["{:.1f}".format(x*1000) for x in ax[0,2].get_xticks()])
        ax[0,3].set_xticklabels(["{:.1f}".format(x*1000) for x in ax[0,3].get_xticks()])
        ax[1,2].set_xticklabels(["{:.1f}".format(x*1000) for x in ax[1,2].get_xticks()])
        ax[1,3].set_xticklabels(["{:.1f}".format(x*1000) for x in ax[1,3].get_xticks()])

        ax[0,3].legend(handles = [L_mix_qc, L_mix_qv, L_mix_qt],
                       labels=[r'ql', r'qv', r'qt'],
                       ncol=3, loc='lower right', bbox_to_anchor=(1, 1), fontsize=8)
        
        Zm_collect = []
        Zc_collect = []
        for axes in ax.flatten()[:-4]:
            xlim = axes.get_xlim()
            Zm_collect.append(axes.plot(xlim, [mix.Zm[0], mix.Zm[0]], '--', lw=1, c='dimgray', zorder=0))
            Zc_collect.append(axes.plot(xlim, [mix.Zc[0], mix.Zc[0]], '--', lw=1, c='dimgray', zorder=0))
            axes.grid()
        
        for axes in ax.flatten()[4:]:
            axes.grid()
            
        twin_ax.set_ylim(ax[0,0].get_ylim())
        twin_ax.set_yticks([])
        twin_ax.spines['left'].set_visible(False)
        twin_ax.spines['right'].set_visible(False)
        twin_ax.tick_params(axis='both', left=False, right=False)
        
        def update(t):
            for axes in ax.flatten():
                    for artist in axes.collections:
                        artist . remove()
                        
            def update_mix(t):
                for Line in Zm_collect:
                    Line[0].set_ydata([mix.Zm[t], mix.Zm[t]])
                for Line in Zc_collect:
                    Line[0].set_ydata([mix.Zc[t], mix.Zc[t]])
                    
                ax[0,3].fill_betweenx(mix.m_zc_3[t, :top_lim+3], 
                                      mix.qv_3[t, :top_lim+3], mix.qv_3[t, :top_lim+3]+mix.qc_3[t, :top_lim+3], 
                                      color='blue', alpha=0.1)
                ax[0,2].fill_betweenx(mix.m_zc_3[t, :top_lim+3], 
                                      0, mix.qc_3[t, :top_lim+3], 
                                      color='blue', alpha=0.1)
                
                L_mix_th.set_data(mix.th_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
                L_mix_Temp.set_data(mix.temp_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
                
                L_mix_th_e.set_data(mix.th_e[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
                L_mix_th_es.set_data(mix.th_es[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
                
                L_mix_qt.set_data(mix.qv_2[t, :top_lim+2], mix.m_zc_2[t, :top_lim+2])
                L_mix_qv.set_data(mix.qv_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
                
                L_mix_qc.set_data(mix.qc_3[t, :top_lim+3], mix.m_zc_3[t, :top_lim+3])
                
            def update_VVM(t):
                
                ax[1,3].fill_betweenx(vvm.m_zc[:top_lim], 
                                      vvm.qv[t, :top_lim], vvm.qv[t, :top_lim]+vvm.qc[t, :top_lim], 
                                      color='blue', alpha=0.1)
                ax[1,2].fill_betweenx(vvm.m_zc[:top_lim], 
                                      0, vvm.qc[t, :top_lim], 
                                      color='blue', alpha=0.1)
                
                L_th.set_xdata(vvm.th[t, :top_lim])
                L_Temp.set_xdata(vvm.temp[t, :top_lim])
                
                L_th_e.set_xdata(vvm.th_e[t, :top_lim])
                L_th_es.set_xdata(vvm.th_es[t, :top_lim])
                
                L_qt.set_xdata(vvm.qv[t, :top_lim]+vvm.qc[t, :top_lim])
                L_qv.set_xdata(vvm.qv[t, :top_lim])
                
                L_qc.set_xdata(vvm.qc[t, :top_lim])
            
            update_mix(t)
            update_VVM(t)
            
            twin_ax.set_yticks([mix.Zm[t], mix.Zc[t]], 
                            labels=[r'$Z_m$ = {:.1f}m'.format(mix.Zm[t]), r'$Z_c$ = {:.1f}m'.format(mix.Zc[t])])
            
            fig.suptitle('TaiwanVVM simulation, tpe20110802cln\nTime: %02d:%1d0 LST @(%8sE, %8sN)' 
                          %(int(t/6), np.mod(t, 6), locate.loc[0], locate.loc[1]), 
                          x=0.1, y=0.975, ha = 'left')
            
            print('\rFrame:{}'.format(t), end='')
            return None
        
        return (FuncAnimation(fig, update, repeat=True,
           frames=range(0, 145, int(144/frames)), interval=10000/frames))

    def skew_T(self, frames):
        
        fig,ax  = plt.subplots(1, 1, figsize = (8, 6), sharey=True)
    
# ==================================================
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))
# ==================================================
# zzz
