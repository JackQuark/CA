# CA5 / mixed layer
# ==================================================
import os
import time
import matplotlib.lines
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
        
        for key, value in kwargs.items():
            if key == 'cache':
            
                self.cache: dict = {
                    'Quark': [120.32586, 23.96824],
                    'Yushan': [120.95952, 23.48761],
                    'Sparrow': [121.01422, 24.82785],
                    'Taipei': [121.51485, 25.03766],
                    'Vincent': [121.61328, 23.97513],
                    'Gary': [121.62970, 24.14743]
                    }
                
                if value in self.cache:
                    self.loc = self.cache[value]
                else:
                    raise ValueError('Invalid cache location')
        
        self.file = f'files/{self.loc[0]}_{self.loc[1]}.nc'  
            
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
        
        self.A  = 2.53e11 # Pa
        self.B  = 5420 # K
        
        self.Gamma_d = self.g / self.Cp # K/m
    
    def th_2_T(self, th, P):
        T = th * (P / 100000)**(self.Rd/self.Cp)
        return T

    def cc_equation(self, T):
        es = 611.2 * np.exp(self.Lv/self.Rv * (1/273 - 1/T)) # Pa
        return es
    
    def cc_AB(self, T):
        es = self.A * np.exp(-self.B / T) # Pa
        return es
        
    def es_2_qvs(self, es, P):
        
        qvs = (self.Rd/self.Rv) * es / P
        return qvs
    
# ==================================================
# VVM data
class VVM:
    
    def __init__(self, file: str):
            
        self.load(file)
        self.Qth  = integrate.cumtrapz(self.wth, dx=600., initial=0) / self.rho[0]
        self.Qqv  = integrate.cumtrapz(self.wqv, dx=600., initial=0) / self.rho[0]
        
        self.temp = thermo.th_2_T(self.th, self.p_zc)
        self.es   = thermo.cc_equation(self.temp)
        self.qvs  = thermo.es_2_qvs(self.es, self.p_zc)
        
    def load(self, file: str):
        rootgrp = nc.Dataset(file)

        topo: int = rootgrp.variables['topo'][0]
        self.m_zc: np.ndarray = rootgrp.variables['m_zc'][topo:]
        self.p_zc: np.ndarray = rootgrp.variables['p_zc'][topo:] * 100 # hPa to Pa
        self.rho : np.ndarray = rootgrp.variables['rho' ][topo:]
        self.th  : np.ndarray = rootgrp.variables['th'  ][:,topo:]
        self.qv  : np.ndarray = rootgrp.variables['qv'  ][:,topo:]
        self.wth : np.ndarray = rootgrp.variables['wth' ][:]
        self.wqv : np.ndarray = rootgrp.variables['wqv' ][:]

        rootgrp.close()

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
    
    def mix_layer(self, Qth, th, Qqv, qv, m_zc):
        
        self.relate_zc = m_zc - m_zc[0]
        self.slope_th = (m_zc[1:] - m_zc[:-1]) / (th[0, 1:] - th[0, :-1])
        self.slope_qv = (m_zc[1:] - m_zc[:-1]) / (qv[0, 1:] - qv[0, :-1])
        self.z_th_ = integrate.cumtrapz(self.relate_zc, th[0, :], initial=0)
        self.z_qv_ = integrate.cumtrapz(qv[0, :], self.relate_zc, initial=0)
        
        # index which mixed layer should insert
        self.lev = np.searchsorted(self.z_th_, Qth, side='right')

        # mix_H = sqrt(2 * Area * slope + z^2) + m_zc[0]
        H_est = self.H_est = np.where(self.lev == 0, m_zc[0],
                             np.sqrt(2 * (Qth - self.z_th_[self.lev-1]) 
                                     * self.slope_th[self.lev-1] 
                                     + self.relate_zc[self.lev-1]**2) 
                             + m_zc[0])
        
        self.H_est_relate = H_est - m_zc[0]
        self.Zmdz = H_est - m_zc[self.lev-1]

        # mix_th = previous level th + delta_H / slope
        self.th_est = np.where(self.H_est_relate == 0., th[0, 0],
                      th[0, self.lev-1] + 
                      self.Zmdz / self.slope_th[self.lev-1])

        # VVM qv at Zm
        self.qv_Zm  = np.where(self.H_est_relate == 0., qv[0, 0], 
                      qv[0, self.lev-1] + 
                      self.Zmdz / self.slope_qv[self.lev-1])
        
        # mix_qv = total area / mix_H
        self.qv_est = np.where(self.H_est_relate == 0., qv[0, 0],
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
            self.m_zc[i, level[i]:level[i]+2] = self.H_est[i]
            self.m_zc[i, level[i]+2:]         = m_zc[level[i]:]
            
            self.p_zc[i, :level[i]]           = p_zc[:level[i]]
            self.p_zc[i, level[i]:level[i]+2] = p_zc[level[i]-1] + P_slope[level[i]-1]\
                                                * (self.H_est[i] - m_zc[level[i]-1])
            self.p_zc[i, level[i]+2:]         = p_zc[level[i]:]
            
            self.th[i, :level[i]+2] = self.th_est[i]
            self.th[i, level[i]+2:] = th[0, level[i]:]
            
            self.qv[i, :level[i]+1] = self.qv_est[i]
            self.qv[i, level[i]+1]  = self.qv_Zm[i]
            self.qv[i, level[i]+2:] = qv[0, level[i]:]
            
            diff_z = np.cumsum(np.append(0, np.diff(self.m_zc[i])))
            
            self.temp[i, :level[i]+2] = thermo.th_2_T(self.th[i][0], self.p_zc[i][0])\
                                        - diff_z[:level[i]+2] * thermo.Gamma_d
            self.temp[i, level[i]+2:] = thermo.th_2_T(th[0, level[i]:], p_zc[level[i]:])
            
            self.temp_d[i, :] = self.temp[i, 0] - diff_z * thermo.Gamma_d
        

    def mix_profile2(self, th, qv, m_zc, p_zc):
        level = np.where(self.lev == 0, 1, self.lev)
        P_slope = (p_zc[1:] - p_zc[:-1]) / (m_zc[1:] - m_zc[:-1])
        mix_zc, mix_p, mix_th, mix_qv, mix_temp, Temp_d = [], [], [], [], [], []
        
        for i in range(145): # [time, level+2]

            mix_zc.append(np.concatenate((m_zc[:level[i]], 2*[self.H_est[i]], 
                                        m_zc[level[i]:])))
            
            mix_p. append(np.concatenate((p_zc[:level[i]], 2*[p_zc[level[i]-1] + P_slope[level[i]-1] * 
                                        (self.H_est[i]-m_zc[level[i]-1])], p_zc[level[i]:])))
            
            mix_th.append(np.concatenate((np.full((level[i]+2), self.th_est[i]),
                                        th[0, level[i]:])))
            
            mix_qv.append(np.concatenate((np.full((level[i]+1), self.qv_est[i]),
                                        [self.qv_Zm[i]], qv[0, level[i]:])))
            
            diff_z = np.cumsum(np.append(0, np.diff(mix_zc[i])))
            
            mix_temp.append(np.concatenate((thermo.th_2_T(mix_th[i][0], mix_p[i][0])
                                            - diff_z[:level[i]+2] * thermo.Gamma_d,
                        thermo.th_2_T(th[0, level[i]:], p_zc[level[i]:]))))
            
            Temp_d.append(mix_temp[i][0] - diff_z * thermo.Gamma_d)
        
        self.m_zc   = np.array(mix_zc)
        self.p_zc   = np.array(mix_p)
        self.th     = np.array(mix_th)
        self.qv     = np.array(mix_qv)
        self.temp   = np.array(mix_temp)
        self.temp_d = np.array(Temp_d)

# ==================================================
# main   
def main():
    global locate, thermo, vvm, mix
    
    locate = Location(cache='Quark')
    thermo = Thermo()
    vvm    = VVM(locate.file)
    mix    = Mix()
    
    def TcZc(qvm, qvs):

        Zc, Tc = [], []
        for i in range(145):
            lev = len(np.where(qvm[i] < qvs[i])[0]) - 1
            if lev < 0:
                Zc.append(np.nan)
                continue
            
            dqv = qvm[i] - qvs[i, lev]
            dz  = dqv * (mix.m_zc[i, lev+1] - mix.m_zc[i, lev]) / (qvs[i, lev+1] - qvs[i, lev])
            Zc.append(mix.m_zc[i, lev] + dz)
        
        return np.array(Tc), np.array(Zc)
    
    global Zc

    Tc, Zc = TcZc(mix.qv[:,0], mix.qvs_d)
    
    Draw('anim', frames=24, save=True)
    
# ==================================================
# Draw
class Draw:
    
    def __init__(self, type: str, save: bool = False, **kwargs) -> None:
        self.fig = plt.figure(figsize=(10, 8))
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
        pass
        
    def animate(self, frames):
        
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot(2, 2, 3)
        ax3 = plt.subplot(2, 2, 4, sharey=ax2)
        ax22= ax2.twinx()
        ax33= ax3.twinx()
        
        self.fig.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3)
        
        top_lim = max(mix.lev) + 5
        
        line_Zc, = ax1.plot(self.t_step, Zc, '-', lw=1, c='b', label='Zc')
        line_Zcm,= ax1.plot(self.t_step, np.where(Zc > mix.H_est, np.nan, Zc), '--', lw=1, c='springgreen', label='Zcm')
        line_Zm, = ax1.plot(self.t_step, mix.H_est, '-', lw=1, c='orange', label='Zm')
        
        ax2.plot(vvm.th[0, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='violet', label=r'Init. $\theta$')
        ax2.plot(vvm.temp[0, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='cyan', label='Init. Temp')
        line_dry1, = ax2.plot([], [], '--', lw=1, c='g', label=r'$\Gamma_d$')
        line_mix_th, = ax2.plot(mix.th[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='r', label=r'Est. $\theta$')
        line_mix_temp, = ax2.plot(mix.temp[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='b', label='Est. Temp')
        line_H1, = ax2.plot(ax2.get_xlim(), np.full((2), mix.H_est[0]), '--', lw=1, c='orange', label=None)
        
        ax3.plot(vvm.qv[0, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='cyan')
        line_dry2, = ax3.plot(mix.qvs_d[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '--', lw=1, c='g', label=r'$\Gamma_d$ qvs')
        line_mix_qv, = ax3.plot(mix.qv[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='b', label='Est. qv')
        line_mix_qvs, = ax3.plot(mix.qvs[0, :top_lim+2], mix.m_zc[0, :top_lim+2], '-', lw=1, c='g', label='Est. qvs')
        line_H2, = ax3.plot(ax3.get_xlim(), np.full((2), mix.H_est[0]), '--', lw=1, c='orange', label='Mixed height')
        dash_Zc, = ax3.plot([], [], '--', c='r', label='Zc', lw=1)
        dot_Zc, = ax3.plot([], [], 'o', c='r', label=None, ms=3)
        
        ax1.set_title('Zm / Zc / Zcm time series')
        ax1.set_xticks(np.linspace(0, 144, 13), labels=np.linspace(0, 24, 13))
        ax1.set_yticks(np.append(ax1.get_yticks(), vvm.m_zc[0]))
        ax1.set_xlim([0, 144])
        ax1.set_ylim([vvm.m_zc[0], ax1.get_ylim()[1]])
        ax1.set_xlabel('Time [LST]')
        ax1.set_ylabel('Height [m]')
        ax1.legend(loc=2)
        ax1.grid()
        
        ax2.set_title(r'$\theta$ / Temp.')
        ax2.set_xlim(line_H1.get_xdata())
        ax2.set_ylim([vvm.m_zc[0], vvm.m_zc[top_lim-1]])
        ax2.set_xlabel('[K]')
        ax2.set_ylabel('Height [m]')
        ax2.legend(loc=3, fontsize=8)
        ax2.grid()
        ax22.set_ylim(ax2.get_ylim())
        
        ax3.set_title(r'qv / qvs')
        ax3.tick_params(axis='y', right=False, left=False)
        ax3.set_xlim(line_H2.get_xdata())
        ax3.set_xlabel('[kg/kg]')
        ax3.legend(fontsize=8)
        ax3.grid()
        ax33.tick_params(axis='y', right=True, left=False)
        ax33.set_ylim(ax2.get_ylim())
        
        def update(i):
            
            line_Zm.set_data(self.t_step[:i+1], mix.H_est[:i+1])
            line_Zc.set_data(self.t_step[:i+1], Zc[:i+1])
            line_Zcm.set_data(self.t_step[:i+1], np.where(Zc[:i+1] > mix.H_est[:i+1], np.nan, Zc[:i+1]))
            
            line_dry1.set_data(mix.temp_d[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_th.set_data(mix.th[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_temp.set_data(mix.temp[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_H1.set_ydata(np.full((2), mix.H_est[i]))
            
            line_dry2.set_data(mix.qvs_d[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_qv.set_data(mix.qv[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_mix_qvs.set_data(mix.qvs[i, :top_lim+2], mix.m_zc[i, :top_lim+2])
            line_H2.set_ydata(np.full((2), mix.H_est[i]))
            dot_Zc.set_data(mix.qv[i, 0], Zc[i])
            dash_Zc.set_data([mix.qv[i, 0], ax3.get_xlim()[1]], [Zc[i], Zc[i]])
            
            ax22.set_yticks([mix.H_est[i]], labels=[f'Zm:{mix.H_est[i]:.1f}'], fontsize=8)
            ax33.set_yticks([Zc[i]], labels=[f'Zc:{Zc[i]:.1f}'], fontsize=8)
            
            self.fig.suptitle('TaiwanVVM simulation, tpe20110802cln\nTime: %02d:%1d0 LST @(%8sE, %8sN)' 
                               %(int(i/6), np.mod(i, 6), locate.loc[0], locate.loc[1]))
            
            print(i)
            return None
    
        return (FuncAnimation(self.fig, update, repeat=True,
           frames=range(0, 145, int(144/frames)), interval=10000/frames))
        
    def saveanim(self, input_anim: FuncAnimation, frames):
        FFwriter = FFMpegWriter(fps = frames/12)
        input_anim.save(f'videos/CA5_{locate.loc[0]}E_{locate.loc[1]}N_{frames}f.mp4', 
                        writer=FFwriter)

# ==================================================
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\ntime : %.3f ms' %((end_time - start_time)*1000))
# ==================================================
# zzz
