# CA5 / mixed layer
# ==================================================
import os
import time
import netCDF4              as nc
import numpy                as np
import matplotlib.pyplot    as plt
from scipy                  import integrate
from matplotlib.animation   import *
from metpy.units            import units
from metpy.plots            import SkewT
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

        self.file = f'../CA5/files/{self.loc[0]:.5f}_{self.loc[1]:.5f}.nc'
            
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
        self.Rd = 287.  # J/kg/K
        self.Cp = 1004. # J/kg/K
        self.Lv = 2.5e6 # J/kg
        self.Rv = 461.  # J/kg/K
        self.g  = 9.81  # m/s^2
        
        self.Gamma_d = self.g / self.Cp # K/m
    
    def th_2_T(self, th, P):
        T = th * (P / 1e5)**(self.Rd/self.Cp)
        return T

    def cc_eq(self, T):
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
        self.es   = thermo.cc_eq(self.temp)
        self.qvs  = thermo.es_2_qvs(self.es, self.p_zc)
        
    def surf_parcel(self):
                
        self.parcel_temp = np.zeros((145, len(self.m_zc)))
        self.parcel_qv   = np.zeros((145, len(self.m_zc)))
        self.parcel_qvs  = np.zeros((145, len(self.m_zc)))
        
        for t in range(145):
            self.parcel_temp[t, 0] = self.temp[t, 0]
            self.parcel_qv[t, 0]   = self.qv[t, 0]
            self.parcel_qvs[t, 0]  = self.qvs[t, 0]
            
            for i in range(len(self.m_zc) - 1):
                
                # saturated
                if self.parcel_qvs[t, i] <= self.parcel_qv[t, i]:
                    tmp_gamma   = thermo.gamma_m(self.parcel_temp[t, i], self.parcel_qvs[t, i])
                    self.parcel_temp[t, i+1] = self.parcel_temp[t, i] - (self.m_zc[i+1] - self.m_zc[i]) * tmp_gamma
                    self.parcel_qvs[t, i+1]  = thermo.es_2_qvs(thermo.cc_eq(self.parcel_temp[t, i+1]), self.p_zc[i+1])
                    self.parcel_qv[t, i+1]   = self.parcel_qvs[t, i+1]
                
                # unsaturated
                else:
                    tmp_gamma   = thermo.Gamma_d
                    self.parcel_temp[t, i+1] = self.parcel_temp[t, i] - (self.m_zc[i+1] - self.m_zc[i]) * tmp_gamma
                    self.parcel_qvs[t, i+1]  = thermo.es_2_qvs(thermo.cc_eq(self.parcel_temp[t, i+1]), self.p_zc[i+1])
                    self.parcel_qv[t, i+1]   = self.parcel_qv[t, i]
        
# ==================================================
# main   
def main():
    global locate, thermo, vvm, mix
    
    locate = Location(name = 'Taipei')
    # locate = Location(id = 'B12209019')
    thermo = Thermo()
    vvm    = VVM(locate.file)
    vvm.surf_parcel()

    # Draw('SkewT', time=72)
    Draw('anim', frames=24, save=True)
    
# ==================================================
# Draw
class Draw:
    
    def __init__(self, type: str, save: bool = False, **kwargs) -> None:
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
                
        elif type == 'profile':
            if 'time' in kwargs:
                self.profile(kwargs['time'])
            else:
                self.profile()
                
        elif type == 'SkewT':
            if 'time' in kwargs:
                self.SkewT(kwargs['time'])
            else:
                self.SkewT()
                
    def profile(self, t:int = 0):
        top_lim = -5
        
        f,ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
        # ax[0].set_facecolor('#D0D0D0')
        ax[0].plot(vvm.temp[t, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='b', label=r'env temp. (VVM)')
        ax[0].plot(vvm.parcel_temp[t, :top_lim], vvm.m_zc[:top_lim], '-', lw=1, c='r', label=r'Parcel temp.')
        ax[0].set_xlabel(r'Temperature ($^\circ$C)')
        ax[0].set_ylabel(r'Height (m)')
        ax[0].set_ylim([vvm.m_zc[0], vvm.m_zc[top_lim-1]])
        ax[0].legend(loc='upper right')
        ax[0].grid()
        
        ax[1].plot(vvm.qv[t, :top_lim], vvm.m_zc[:top_lim], '--', lw=1, c='b', label=r'env qv (VVM)')
        ax[1].plot(vvm.parcel_qv[t, :top_lim], vvm.m_zc[:top_lim], '-', lw=1, c='r', label=r'Parcel qv')
        ax[1].set_xlabel(r'qv (kg/kg)')
        ax[1].legend(loc='upper right')
        ax[1].grid()
        
        f.suptitle('TaiwanVVM simulation, tpe20110802cln\nTime: %02d:%1d0 LST @(%8sE, %8sN)' 
                               %(int(t/6), np.mod(t, 6), locate.loc[0], locate.loc[1]))
        plt.show()
        
    def SkewT(self, t:int = 0):
        vvm_Temp = vvm.temp[t, :] - 273.15
        vvm_P = vvm.p_zc[:] / 100
        parcel_Temp = vvm.parcel_temp[t, :] - 273.15
        
        fig  = plt.figure(figsize=(6, 6))
        # fig.subplots_adjust(bottom=0.15, top=0.95, left=0.2, right=0.8)
        
        skew = SkewT(fig, rotation=45, aspect=100.)
        skew.plot(vvm_P, vvm_Temp, 'b', lw=1)
        skew.plot(vvm_P, parcel_Temp, 'k', lw=0.75)
        
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        
        tmp_xarr = np.arange(-200, 41, 20)
        size = np.size(vvm_P)
        for i in range(np.size(tmp_xarr)-1):
            skew.shade_area(vvm_P, np.full((size), tmp_xarr[i]), np.full((size), tmp_xarr[i] + 10), 
                            alpha=0.2, color='green')
            
        skew.plot_dry_adiabats(ls='-', lw=0.5, colors='peru')
        skew.plot_moist_adiabats(ls=':', lw=0.5, colors='green')
        skew.plot_mixing_lines(ls='--', lw=0.5, colors='green')
        
        skew.ax.set_ylabel('Pressure [hPa]')
        skew.ax.set_title('Skew-T')
        
        fig.suptitle(r'SKEW T, log p DIAGRAM', y = 0.05, fontsize=12)
        plt.show()
           
    def animate(self, frames):
        
        vvm_Temp = vvm.temp[0, :] - 273.15
        vvm_P = vvm.p_zc[:] / 100
        parcel_Temp = vvm.parcel_temp[0, :] - 273.15
        
        fig  = plt.figure(figsize=(6, 6))
        # fig.subplots_adjust(bottom=0.15, top=0.95, left=0.2, right=0.8)
        
        skew = SkewT(fig, rotation=45, aspect=100.)
        skew.plot(vvm_P, vvm_Temp, 'b', lw=1)
        skew.plot(vvm_P, parcel_Temp, 'k', lw=0.75)
        
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        
        tmp_xarr = np.arange(-200, 41, 20)
        size = np.size(vvm_P)
        for i in range(np.size(tmp_xarr)-1):
            skew.shade_area(vvm_P, np.full((size), tmp_xarr[i]), np.full((size), tmp_xarr[i] + 10), 
                            alpha=0.2, color='green')
            
        skew.plot_dry_adiabats(ls='-', lw=0.5, colors='peru')
        skew.plot_moist_adiabats(ls=':', lw=0.5, colors='green')
        skew.plot_mixing_lines(ls='--', lw=0.5, colors='green')
        
        skew.ax.set_ylabel('Pressure [hPa]')
        skew.ax.set_title('Skew-T')
        
        def update(i):
            
            self.fig.suptitle('TaiwanVVM simulation, tpe20110802cln @(%8sE, %8sN)\
                \nEnv. and Parcel Temp./qv/qvs profile, Time: %02d:%1d0 LST' 
                               %(locate.loc[0], locate.loc[1], int(i/6), np.mod(i, 6)), 
                               x=0.02, y=0.98, ha = 'left')
            
            print('\rFrame:{}'.format(i), end='')
            return None
    
        return (FuncAnimation(self.fig, update, repeat=True,
           frames=range(0, 145, int(144/frames)), interval=10000/frames))
        
    def saveanim(self, input_anim: FuncAnimation, frames):
        FFwriter = FFMpegWriter(fps = frames/12)
        input_anim.save(f'videos/SkewT_{locate.file_name}_{frames}f.mp4', 
                        writer=FFwriter, dpi=200)

# ==================================================
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))
# ==================================================
# hao yeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
