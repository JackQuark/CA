# CA5 / mixed layer
# ==================================================
import os
import time
import netCDF4              as nc
import numpy                as np
import sympy                as sym
import scipy                as sci
from scipy      import integrate
import matplotlib.pyplot    as plt
from matplotlib.animation import *
# ==================================================
# location info
class Location:

    def __init__(self, **kwargs):

        if 'id' in kwargs:
            self.load()
                
            if kwargs['id'] in self.id:
                idx = int(np.where(self.id == kwargs['id'])[0])
                self.loc = [self.lon[idx], self.lat[idx]]
                self.file_name = self.id[idx]
            else:
                raise ValueError('Invalid ID')
        
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

        self.file = f'files/{self.loc[0]:.5f}_{self.loc[1]:.5f}.nc'
            
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
        self.Rd = 287. # J/kg/K
        self.Cp = 1004. # J/kg/K
        self.Lv = 2.5e6 # J/kg
        self.Rv = 461. # J/kg/K
        self.g  = 9.81 # m/s^2
        
        self.Gamma_d = self.g / self.Cp # K/m
    
    def th_2_T(self, th, P):
        T = th * (P / 1e5)**(self.Rd/self.Cp)
        return T

    def cc_equation(self, T):
        es = 611.2 * np.exp(self.Lv/self.Rv * (1/273 - 1/T)) # Pa
        return es
        
    def es_2_qvs(self, es, P):
        qvs = (self.Rd/self.Rv) * es / P
        return qvs

    def Cp_star(self, T, qvs):
        Cp_star = (self.Cp *
            (1 + (self.Lv**2 * qvs / (self.Cp * self.Rv * T**2))) /
            (1 + (self.Lv * qvs / (self.Rd * T))))
        return Cp_star
    
    def gamma_m(self, T, qvs):
        return self.g / self.Cp_star(T, qvs)
    
# ==================================================
# VVM data
class VVM:
    
    def __init__(self, file: str):
        
        with nc.Dataset(file) as rootgrp:

            topo: int = rootgrp.variables['topo'][0]
            self.m_zc: np.ndarray = rootgrp.variables['m_zc'][topo:]
            self.p_zc: np.ndarray = rootgrp.variables['p_zc'][topo:] * 100 # hPa to Pa
            self.rho : np.ndarray = rootgrp.variables['rho' ][topo:]
            self.th  : np.ndarray = rootgrp.variables['th'  ][:,topo:]
            self.qv  : np.ndarray = rootgrp.variables['qv'  ][:,topo:]
            self.wth : np.ndarray = rootgrp.variables['wth' ][:]
            self.wqv : np.ndarray = rootgrp.variables['wqv' ][:]
            
        self.Qth  = integrate.cumtrapz(self.wth, dx=600., initial=0) / self.rho[0]
        self.Qqv  = integrate.cumtrapz(self.wqv, dx=600., initial=0) / self.rho[0]
        self.temp = thermo.th_2_T(self.th, self.p_zc)
        self.es   = thermo.cc_equation(self.temp)
        self.qvs  = thermo.es_2_qvs(self.es, self.p_zc)
        self.SH_flux = self.wth * thermo.Cp * (self.p_zc[0] / 1e5)**(0.286)
        
# ==================================================
# Mixed data
class Mix:

    def __init__(self,):
        self.mix_layer(vvm.Qth, vvm.th, vvm.Qqv, vvm.qv, vvm.m_zc)
        self.mix_profile(vvm.th, vvm.qv, vvm.m_zc, vvm.p_zc)
        
        self.es    = thermo.cc_equation(self.temp)
        self.qvs   = thermo.es_2_qvs(self.es, self.p_zc)
        self.es_d  = thermo.cc_equation(self.temp_d)
        self.qvs_d = thermo.es_2_qvs(self.es_d, self.p_zc)
        
        def TcZc(qvm, qvs):
        
            Zc = []
            for i in range(145):
                lev = len(np.where(qvm[i] < qvs[i])[0]) - 1
                
                dqv = qvm[i] - qvs[i, lev]
                dz  = dqv * (mix.m_zc[i, lev+1] - mix.m_zc[i, lev]) / (qvs[i, lev+1] - qvs[i, lev])
                Zc.append(mix.m_zc[i, lev] + dz)
        
        self.Zc = np.zeros(145)
        for t in range(145):
            lev = len(np.where(self.qvm[t] < self.qvs_d[t])[0]) - 1
            
            dqv = self.qvm[t] - self.qvs_d[t, lev]
            
            if (self.m_zc[t, lev+1] - self.m_zc[t, lev]) == 0:
                dz = 0.
            else:
                dz = dqv * (self.m_zc[t, lev+1] - self.m_zc[t, lev]) / (self.qvs_d[t, lev+1] - self.qvs_d[t, lev])
            
            self.Zc[t] = self.m_zc[t, lev] + dz
    
    def mix_layer(self, Qth, th, Qqv, qv, m_zc):
        
        self.relate_zc = m_zc - m_zc[0]
        self.slope_th = (m_zc[1:] - m_zc[:-1]) / (th[0, 1:] - th[0, :-1])
        self.slope_qv = (m_zc[1:] - m_zc[:-1]) / (qv[0, 1:] - qv[0, :-1])
        self.z_th_ = integrate.cumtrapz(self.relate_zc, th[0, :], initial=0)
        self.z_qv_ = integrate.cumtrapz(qv[0, :], self.relate_zc, initial=0)
        # index which mixed layer should insert
        self.lev = np.searchsorted(self.z_th_, Qth, side='right')

        # mix_H = sqrt(2 * Area * slope + z^2) + m_zc[0]
        self.Zm = np.where(self.lev == 0, m_zc[0],
                             np.sqrt(2 * (Qth - self.z_th_[self.lev-1]) 
                                     * self.slope_th[self.lev-1] 
                                     + self.relate_zc[self.lev-1]**2) 
                             + m_zc[0])
        
        self.H_est_relate = self.Zm - m_zc[0]
        self.Zmdz = self.Zm - m_zc[self.lev-1]

        # mix_th = previous level th + delta_H / slope
        self.thm    = np.where(self.H_est_relate == 0., th[0, 0],
                      th[0, self.lev-1] + 
                      self.Zmdz / self.slope_th[self.lev-1])

        # VVM qv at Zm
        self.qv_Zm  = np.where(self.H_est_relate == 0., qv[0, 0], 
                      qv[0, self.lev-1] + 
                      self.Zmdz / self.slope_qv[self.lev-1])
        
        # mix_qv = total area / mix_H
        self.qvm    = np.where(self.H_est_relate == 0., qv[0, 0],
                      (Qqv + self.z_qv_[self.lev-1] + 
                      (qv[0, self.lev-1] + self.qv_Zm) * self.Zmdz / 2)
                      / self.H_est_relate)
        
    # insert 2 points at Zm
    def mix_profile(self, th, qv, m_zc, p_zc):
        level    = np.where(self.lev == 0, 1, self.lev)
        
        P_slope  = (p_zc[1:] - p_zc[:-1]) / (m_zc[1:] - m_zc[:-1])
        self.m_zc   = np.zeros((145, len(m_zc)+2))
        self.p_zc   = np.zeros((145, len(m_zc)+2))
        self.th     = np.zeros((145, len(m_zc)+2))
        self.qv     = np.zeros((145, len(m_zc)+2))
        self.temp   = np.zeros((145, len(m_zc)+2))
        self.temp_d = np.zeros((145, len(m_zc)+2))
        
        for i in range(145): # [time, level+2]
            self.m_zc[i, :level[i]]           = m_zc[:level[i]]
            self.m_zc[i, level[i]:level[i]+2] = self.Zm[i]
            self.m_zc[i, level[i]+2:]         = m_zc[level[i]:]
            
            self.p_zc[i, :level[i]]           = p_zc[:level[i]]
            self.p_zc[i, level[i]:level[i]+2] = p_zc[level[i]-1] + (P_slope[level[i]-1]
                                                * (self.Zm[i] - m_zc[level[i]-1]))
            self.p_zc[i, level[i]+2:]         = p_zc[level[i]:]
            
            self.th[i, :level[i]+2] = self.thm[i]
            self.th[i, level[i]+2:] = th[0, level[i]:]
            
            self.qv[i, :level[i]+1] = self.qvm[i]
            self.qv[i, level[i]+1]  = self.qv_Zm[i]
            self.qv[i, level[i]+2:] = qv[0, level[i]:]
            
            diff_z = np.cumsum(np.append(0, np.diff(self.m_zc[i])))
            
            self.temp[i, :level[i]+2] = thermo.th_2_T(self.th[i][0], self.p_zc[i][0])\
                                        - diff_z[:level[i]+2] * thermo.Gamma_d
            self.temp[i, level[i]+2:] = thermo.th_2_T(th[0, level[i]:], p_zc[level[i]:])
            
            self.temp_d[i, :] = thermo.th_2_T(self.thm[i], self.p_zc[i])
            # self.temp_d[i, :] = self.temp[i, 0] - diff_z * thermo.Gamma_d
        
# ==================================================
# main   
def main():
    global locate, thermo, vvm, mix
    
    locate = Location(name = 'Quark')
    # locate = Location(id = 'B12209044')
    thermo = Thermo()
    vvm    = VVM(locate.file)
    mix    = Mix()
    
    def TcZc(qvm, qvs):
        
        Zc, = [],
        for i in range(145):
            lev = len(np.where(qvm[i] < qvs[i])[0]) - 1

            dqv = qvm[i] - qvs[i, lev]
            dz  = dqv * (mix.m_zc[i, lev+1] - mix.m_zc[i, lev]) / (qvs[i, lev+1] - qvs[i, lev])
            Zc.append(mix.m_zc[i, lev] + dz)
        
        return np.array(Zc)
    
    global Zc
    Zc = TcZc(mix.qv[:,0], mix.qvs_d)
    
    def test(*args):
        for arr in args:
            plt.plot(arr)
        plt.legend()
        plt.show()
    
    # test(mix.qvs_d[:,0], mix.qvm)
    #Draw('vertical_test')
    Draw('anim', frames=12, save=True)
    
# ==================================================
# Draw
class Draw:
    
    def __init__(self, type: str, save: bool = False, **kwargs) -> None:
        self.fig = plt.figure(figsize=(8, 6))
        self.t_step = np.linspace(0, 144, 145)
        
        if type == 'anim':
            if 'frames' in kwargs:
                self.frames = kwargs['frames']
            else:
                self.frames = 24
            
            global anim
            anim = self.animate(self.frames)
            
            if save:
                self.saveanim(anim, self.frames)
                
        elif type == 'vertical_test':
            self.plot()
    
    def plot(self):
        
        ax0 = plt.subplot(2, 1, 1)
        ax1 = plt.subplot(2, 1, 2)
        pass
        
    def animate(self, frames):
        
        ax_th  = plt.subplot(2, 2, 1)
        ax_qv  = plt.subplot(2, 2, 2)
        ax_wth = plt.subplot(3, 2, 5)
        ax_wqv = plt.subplot(3, 2, 6)
        ax22   = ax_th.twinx()
        ax33   = ax_qv.twinx()
        ax_wqv.sharex(ax_wth)
        
        self.fig.subplots_adjust(bottom=0.1, top=0.75, hspace=0.3, wspace=0.25,
                                 left=0.1, right=0.9,)
    
        top_lim = max(mix.lev) + 5
        
        th_bbox  = ax_th.get_position()
        qv_bbox  = ax_qv.get_position()

        ax_th.set_position([th_bbox.x0, th_bbox.y0-0.025, th_bbox.width, th_bbox.height*1.2])
        ax_qv.set_position([qv_bbox.x0, qv_bbox.y0-0.025, qv_bbox.width, qv_bbox.height*1.2])
            
        # init
        # =====sub_th=====
        ax_th.plot(vvm.th[0, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='violet', label=r'Init. $\theta$')
        ax_th.plot(vvm.temp[0, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='cyan', label='Init. Temp')
        line_dry1, = ax_th.plot([], [], '--', lw=1, c='g', label=r'$\Gamma_d$')
        line_mix_th, = ax_th.plot(mix.th[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='r', label=r'Est. $\theta$')
        line_mix_temp, = ax_th.plot(mix.temp[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='b', label='Est. Temp')
        dash_Zm1, = ax_th.plot(ax_th.get_xlim(), np.full((2), mix.Zm[0]), '--', lw=1, c='orange', label='Zm')
        # =====sub_qv=====
        ax_qv.plot(vvm.qv[0, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='cyan', label='Init. qv')
        line_mix_qv, = ax_qv.plot(mix.qv[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='b', label='Est. qv')
        line_mix_qvs, = ax_qv.plot(mix.qvs[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='g', label='Est. qvs')
        line_dry2, = ax_qv.plot(mix.qvs_d[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '--', lw=1, c='g', label=r'$\Gamma_d$ qvs')
        dash_Zm2, = ax_qv.plot(ax_qv.get_xlim(), np.full((2), mix.Zm[0]), '--', lw=1, c='orange', label='Zm')
        dash_Zc, = ax_qv.plot([], [], '--', c='r', label='Zc', lw=1)
        dot_Zc, = ax_qv.plot([], [], 'o', c='r', label=None, ms=3)
        # =====sub_wth=====
        ax_wth.plot(self.t_step, vvm.SH_flux, '-', lw=1, c='r', label='wth')
        ax_wth.plot(self.t_step, np.full((len(self.t_step)), 0), '--', lw=0.8, c='k', label=None)
        # =====sub_wqv=====
        ax_wqv.plot(self.t_step, vvm.wqv*1e3, '-', lw=1, c='b', label='wqv')
        ax_wqv.plot(self.t_step, np.full((len(self.t_step)), 0), '--', lw=0.8, c='k', label=None)
        
        def ytick_(y_min, y_max):
            step=100
            while len(np.arange(y_min, y_max, step)) > 6:
                step += 100
            return np.arange(y_min, y_max, step)
        
        ytick = ytick_(vvm.m_zc[0], vvm.m_zc[top_lim])
        
        # setting
        # =====sub_th=====
        ax_th.set_title(r'$\theta$ / Temp.', loc='left')
        ax_th.set_yticks(ytick)
        ax_th.set_xlim(dash_Zm1.get_xdata())
        ax_th.set_ylim([vvm.m_zc[0], vvm.m_zc[top_lim-1]])
        ax_th.set_xlabel('[K]')
        ax_th.set_ylabel('Height [m]')
        ax_th.legend(loc='lower right', bbox_to_anchor=(1, 1), 
                    ncols=2, handlelength=1, fontsize=8)
        ax_th.grid()
        ax22.set_ylim(ax_th.get_ylim())
        # =====sub_qv=====
        ax_qv.set_title(r'qv / qvs', loc='left')
        ax_qv.tick_params(axis='y', right=False, left=False)
        ax_qv.set_yticks(ax_th.get_yticks())
        ax_qv.set_xlim(dash_Zm2.get_xdata())
        ax_qv.set_ylim([vvm.m_zc[0], vvm.m_zc[top_lim-1]])
        ax_qv.set_yticklabels([])
        ax_qv.set_xlabel('[kg/kg]')
        ax_qv.legend(loc='lower right', bbox_to_anchor=(1, 1), 
                    ncols=2, handlelength=1, fontsize=8)
        ax_qv.grid()
        ax33.tick_params(axis='y', right=True, left=False)
        ax33.set_ylim(ax_th.get_ylim())
        # =====sub_wth=====
        ax_wth.set_title(r'Sensible heat flux [$W{\cdot}m^{-2}$]', loc='left')
        ax_wth.set_xticks(np.arange(0, 144.1, 24), labels=np.arange(0, 24.1, 4, dtype=int))
        ax_wth.set_xlim([0, 144])
        ax_wth.set_xlabel('Time [h]')
        ax_wth.grid()
        # =====sub_wqv=====
        ax_wqv.set_title(r'Surface flux of qv [$g{\cdot}m^{-2}{\cdot}s^{-1}$]', loc='left')
        ax_wqv.set_xlabel('Time [h]')
        ax_wqv.grid()
        
        def update(i):
            
            for ax in [ax_wth, ax_wqv]:
                for artist in ax.collections:
                    artist . remove()
            
            ax_wth.fill_between(np.arange(0, i+1, 1), 0, vvm.SH_flux[:i+1], color='red', alpha=0.2)
            ax_wqv.fill_between(np.arange(0, i+1, 1), 0, vvm.wqv[:i+1]*1e3, color='blue', alpha=0.2)
            
            # =====sub_th=====
            line_dry1.set_data(mix.temp_d[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_th.set_data(mix.th[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_temp.set_data(mix.temp[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            dash_Zm1.set_ydata(np.full((2), mix.Zm[i]))
            # =====sub_qv=====
            line_dry2.set_data(mix.qvs_d[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_qv.set_data(mix.qv[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_qvs.set_data(mix.qvs[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            dash_Zm2.set_ydata(np.full((2), mix.Zm[i]))
            dot_Zc.set_data(mix.qv[i, 0], Zc[i])
            dash_Zc.set_data([mix.qv[i, 0], ax_qv.get_xlim()[1]], [Zc[i], Zc[i]])
            
            ax22.set_yticks([mix.Zm[i]], labels=[f'{mix.Zm[i]:.1f}m'], fontsize=8)
            ax33.set_yticks([Zc[i]], labels=[f'{Zc[i]:.1f}m'], fontsize=8)
            
            self.fig.suptitle('TaiwanVVM simulation, tpe20110802cln\nTime: %02d:%1d0 LST @(%8sE, %8sN)' 
                               %(int(i/6), np.mod(i, 6), locate.loc[0], locate.loc[1]), 
                               x=0.02, y=0.98, ha = 'left')
            
            print('\rFrame:{}'.format(i), end='')
            return None
    
        return (FuncAnimation(self.fig, update, repeat=True,
           frames=range(0, 145, int(144/frames)), interval=10000/frames))
        
    def saveanim(self, input_anim: FuncAnimation, frames):
        FFwriter = FFMpegWriter(fps = frames/12)
        input_anim.save(f'videos/CA5_1_{locate.file_name}_{frames}f.mp4', 
                        writer=FFwriter, dpi=200)

# ==================================================
if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))
# ==================================================
# zzz
