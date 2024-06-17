# vpython test
# pendulum wave
# ==================================================
import numpy as np
from time import time
from vpython import *
# ==================================================
g = 9.81
n_balls = 8
class Pendulum:
    
    def __init__(self):
        self.scene = canvas(width = 600, height = 600)
        self.balls = [sphere(pos = vec(0,0,j*-0.5),radius = 0.2) for j in range(n_balls)]
        self.lines = [curve(pos = [(0,0,0),(1,0,0)], radius=0.01) for _ in range(n_balls)]
        self.length = np.linspace(1, 4.5, n_balls)
        
        self.reset()

    def __call__(self, dt):
        x, y = self.midpoint_method(dt)
        self.update_loc(x, y)
    
    def reset(self):
        self.theta = np.full(n_balls, np.pi / 4)
        self.omega = np.zeros(n_balls)
    
    def midpoint_method(self, dt):
        y = -self.length * np.cos(self.theta)
        x =  self.length * np.sin(self.theta)
        theta_half = self.theta + 0.5 * self.omega * dt
        omega_half = self.omega - (g / self.length) * np.sin(self.theta) * dt * 0.5
        self.theta = self.theta + omega_half * dt
        self.omega = self.omega - (g / self.length) * np.sin(theta_half) * dt
        
        return x, y
        
    def update_loc(self, x, y):
        for n in range(n_balls):
            self.balls[n].pos.y = y[n]
            self.balls[n].pos.x = x[n]
            self.lines[n].modify(0, pos=vec(0, 0, n*-0.5))
            self.lines[n].modify(1, pos=vec(x[n], y[n], n*-0.5))
    
class Opreter:
    def __init__(self, scene: canvas):
        button(text='Pause', pos=scene.title_anchor, bind=self.runma)
        button(text='Reset', pos=scene.title_anchor, bind=self.reset)
        button(text='Stop',  pos=scene.title_anchor, bind=self.stop)
        
        self.running = True
        self.dt = 0.01
    
    def runma(self, b: button):
        self.running = not self.running
        
        if self.running:
            b.text = 'Pause'
            self.dt = 0.01    
        else:
            b.text = 'Run'
            self.dt = 0
        
    def reset(self, b: button):
        pendulum.reset()

    def stop(self, b: button):
        self.running = None
    
# ==================================================

def main():
    global pendulum, opreter
    
    pendulum = Pendulum()
    opreter  = Opreter(pendulum.scene)
    
    while not opreter.running is None:
        rate(100)
        pendulum(opreter.dt)
    
    return None

# ==================================================
# owowoowo
if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))