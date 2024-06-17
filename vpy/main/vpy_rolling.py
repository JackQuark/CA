# vpython test
# rolling ball
# ==================================================
import numpy as np
from time    import perf_counter
from vpython import *
# ==================================================
"""
vpython coordinate:
  | y
  |
  |__ __ __ x
 /
/z
    _
 __(.)< quack 
 \=__)
'"'"'"'"'"'"'"
"""
# ==================================================
class Env(object):
    fps: int   = 100
    dt : float = 0.01
    g  : float = 9.81
    
    def __init__(self):
        self.scene = canvas(width = 900, height = 600)
        self.scene.caption = \
        """
        'w': forward, 'a': left, 's': backward, 'd': right
        The direction depends on the perspective.
        """
        self.wallB = box(pos=vector(0, -0.55, 0), size=vector(20, 0.1, 20),
                         texture=textures.wood)
        
        # self.base_plane = box(pos=vector(0, 0, 0), size=vector(25, 0.01, 25), 
        #                       color=color.gray(0.5))
    @ staticmethod
    def gravity(v):
        return v + vector(0, -Env.g * Env.dt, 0)
    
    def reset(self):
        for items in self.items.values():
            items.reset()

    def perspective(self) -> float:
        return np.arctan2(self.scene.forward.z, self.scene.forward.x)
    
    def camera_follow(self, position: vector):
        try:
            self.scene.center = position
        except:
            pass
        
class Phys_ball(object):
    def __init__(self, x, y, z, radius):
        self.ball = sphere(pos=vector(x, y, z), radius=radius, texture=textures.earth)
        
        self._velocity  = vector(0, 0, 0)
        self.grad_v_indx = {'w': 0, 's': 0, 'a': 0, 'd': 0}
        self.ralative_speed = {'w': 0, 's': 0, 'a': 0, 'd': 0}
        
    class state(object):
        move = False
        fly  = False
        
    @ property
    def velocity(self):
        return self._velocity
    
    @ velocity.setter
    def velocity(self, value):
        self._velocity = value

    def speed_update(self, sight_angle):
        self.ralative_speed['w'] = vector(  np.cos(sight_angle), 0,  np.sin(sight_angle))
        self.ralative_speed['s'] = vector( -np.cos(sight_angle), 0, -np.sin(sight_angle))
        self.ralative_speed['a'] = vector(  np.cos(sight_angle-np.pi/2), 0,  np.sin(sight_angle-np.pi/2))
        self.ralative_speed['d'] = vector(  np.cos(sight_angle+np.pi/2), 0,  np.sin(sight_angle+np.pi/2))
    
    def gradient_func(self, x, ratios=1):
        return (2*x**2 - x**4) * ratios
    
    def gradient_v(self, *keys):
        
        wasd = [x for x in keys if x in ['w', 'a', 's', 'd']]
        
        for key in wasd:
            if self.grad_v_indx[key] < 1:
                self.grad_v_indx[key] += 0.01
        
        tmp = vector(0, self.velocity.y, 0)
        for x in self.grad_v_indx.keys():
            if self.grad_v_indx[x] > 0:
                self.grad_v_indx[x] -= 0.005

            tmp += self.ralative_speed[x] * \
                self.gradient_func(self.grad_v_indx[x], 3)
        
        if ' ' in keys and not self.state.fly:
            tmp.y += 3
        
        self.velocity = tmp
        
    def update(self, *cmd):
        
        self.speed_update(env.perspective())
        self.gradient_v(*cmd)
        
        if not self.ball.pos.y <= 0:
            self.velocity = Env.gravity(self.velocity)
            self.state.fly = True
            
        elif self.ball.pos.y <= 0 and self.state.fly:
            self.velocity.y = 0
            self.ball.pos.y = 0
            self.state.fly = False
        
        ds = self.velocity * Env.dt
        self.ball.pos += ds
        self.ball.rotate(angle=ds.cross(vector(0, -1, 0)).mag / self.ball.radius, 
                         axis=ds.cross(vector(0, -1, 0)))
    
    def reset(self):
        self.ball.pos = vector(0, 0, 0)
        self.velocity = vector(0, 0, 0)
        self.grad_v_indx = {'w': 0, 's': 0, 'a': 0, 'd': 0}
        self.ralative_speed = {'w': 0, 's': 0, 'a': 0, 'd': 0}

class Game(object):
    state = True
    
    def __init__(self, game_env: Env):
        self.game_env = game_env
        
        button(text='Reset', pos=game_env.scene.title_anchor, 
               bind=lambda: game_env.reset())
        button(text='Stop',  pos=game_env.scene.title_anchor, 
               bind=lambda: setattr(Game, 'state', False))
        
        self.ball_main = Phys_ball(0, 0, 0, 0.5)
        
        game_env.items = {'ball_main': self.ball_main}
        
        self.valid_key = ['w', 'a', 'd','s', ' ', 'p']
        self.kb_state = {'w': False, 'a': False, 'd': False,'s': False, ' ': False}
        game_env.scene.bind('keydown', self.kb_down_cmd)
        game_env.scene.bind('keyup',   self.kd_up_cmd)
        
    # ========== keyboard ==========
    def kb_down_cmd(self, event):
        key = event.key
        
        if key in self.valid_key:
            if key == 'p': # scene info
                print("fov:"        + str(self.game_env.scene.fov))
                print("camera.pos:" + str(self.game_env.scene.camera.pos))
                print("center:"     + str(self.game_env.scene.center))
                print("forward:"    + str(self.game_env.scene.forward))
                print("range:"      + str(self.game_env.scene.range))
                print("==============================")
            
            elif not self.kb_state[key]:
                self.kb_state[key] = True
    
    def kd_up_cmd(self, event):
        key = event.key
        
        if key in self.valid_key:
            if self.kb_state[key]:
                self.kb_state[key] = False
    
    # ========== total update ==========
    def execute(self):
        self.ball_main.update(*[k for k, v in self.kb_state.items() if v])
        self.game_env.camera_follow(self.ball_main.ball.pos)
        
        args = [str(x) for x in self.kb_state.items()]
        print('{:<12s}, {:<12s}, {:<12s}, {:<12s}, {:<12s}\r'.format(*args), end='')
    
# ==================================================
# Main
def main():
    global env, game

    env = Env()
    game = Game(env)
    
    while Game.state:
        rate(Env.fps)
        game.execute()

# ==================================================
# exe
if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.4f s' %(end_time - start_time))
# ==================================================