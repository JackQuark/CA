# vpython test
# rolling ball
# ==================================================
import numpy as np
from time    import perf_counter
from vpython import *
# ==================================================
"""
vpython coordinate:
  y|
   |
   |__ __ __x
  /
z/
    _
 __(.)< quack
 \=__)
'"'"'"'"'"'"'"
"""
# ==================================================
class Env(object):
    fps: int   = 100  # frame per second
    dt : float = 0.01 # s
    g  : float = 9.81 # m/s^2
    
    def __init__(self):
        self.scene = canvas(width = 900, height = 600)
        # self.wallB = box(pos=vector(0., -0.5, 0.), size=vector(20., 0.001, 20.),
        #                  texture=textures.wood)
        
        self.items = {}
        self.inherit_from_level(level_0)
        
    def inherit_from_level(self, Level):
        self.level = Level()
        pass

    @ staticmethod
    def gravity(v) -> vector:
        return v + vector(0, -Env.g * Env.dt, 0)
    
    def normal_force(v, angle: float) -> vector:
        return
        
    def reset(self):
        for item in self.items.values():
            item.reset()

    def perspective(self) -> float:
        return np.arctan2(self.scene.forward.z, self.scene.forward.x)
    
    def camera_follow(self, position: vector):
        try:
            self.scene.center = position
        except:
            pass
        
class level_0(object):
    def __init__(self):
        # obstacles init
        self.road_1 = Phys_box(vector(0, 0, 5),  vector(2, 0.001, 10), vector(0, 1, 0), 
                               _texture=textures.wood)
        self.road_2 = Phys_box(vector(4, 0, 11), vector(10, 0.001, 2), vector(0, 1, 0), 
                               _texture=textures.wood)
        self.road_3 = Phys_box(vector(10, 0, 7), vector(2, 0.001, 10), vector(0, 1, 0), 
                               _texture=textures.wood)
        pass

class Phys_ball(sphere):
    def __init__(self, x, y, z, radius, _texture=None):
        super().__init__(pos=vector(x, y, z), radius=radius, texture=_texture)
        self._mass = 1
        self._velocity  = vector(0, 0, 0)
        self._acceleration = vector(0, 0, 0)
        self.kb_v_ratio = {'w': 0, 's': 0, 'a': 0, 'd': 0}
        self.kb_dir = {'w': 0, 's': 0, 'a': 0, 'd': 0}
        
    class state(object):
        # state of ball
        move: bool = False
        fly: bool = False
        
    @ property
    def velocity(self):
        return self._velocity
    @ velocity.setter
    def velocity(self, value):
        self._velocity = value
    
    # ========== keyboard input ==========
    def kb_dir_update(self, theta):
        # direction of keyboard input
        # depends on perspective (theta).
        # 'w': vector(cos, 0, sin)
        self.kb_dir['w'] = vector(  np.cos(theta), 0,  np.sin(theta))
        self.kb_dir['s'] = vector( -np.cos(theta), 0, -np.sin(theta))
        self.kb_dir['a'] = vector(  np.cos(theta-np.pi/2), 0,  np.sin(theta-np.pi/2))
        self.kb_dir['d'] = vector(  np.cos(theta+np.pi/2), 0,  np.sin(theta+np.pi/2))
    
    def kb_damping_func(self, x, ratios=1):
        # velocity damping function
        # x: ratio of velocity to keyboard input (0~1)
        # ratios: maximum velocity = ratios
        return (2*x**2 - x**4) * ratios  
    
    def kb_v_update(self, *event):
        # w a s d 
        for key in self.kb_v_ratio.keys():
            # accelerate if kb input
            if key in event and self.kb_v_ratio[key] < 1:
                self.kb_v_ratio[key] += Env.dt
            # slow down if not kb input
            elif not self.state.fly and self.kb_v_ratio[key] > 0:
                self.kb_v_ratio[key] -= Env.dt
        
        tmp = vector(0, self.velocity.y, 0)
        for x in self.kb_v_ratio.keys():
            tmp += self.kb_dir[x] * \
                self.kb_damping_func(self.kb_v_ratio[x], 5)
        
        # ' ' jump
        if ' ' in event and not self.state.fly:
            tmp.y += 3
        
        self.velocity = tmp  
    
    # ========== env interaction ==========
    def domain_check(self):
        
        pass
    
    # ========== update ==========
    def update(self, *cmd):
    
        self.kb_dir_update(env.perspective())
        self.kb_v_update(*cmd)
        
        if self.state.fly:
            self.velocity = Env.gravity(self.velocity)
        
        if not self.pos.y <= 0:
            self.velocity = Env.gravity(self.velocity)
            self.state.fly = True
            
        elif self.pos.y <= 0 and self.state.fly:
            self.velocity.y = 0
            self.pos.y = 0
            self.state.fly = False
        
        ds = self.velocity * Env.dt
        self.pos += ds
        self.rotate(angle=ds.cross(vector(0, -1, 0)).mag / self.radius, 
                         axis=ds.cross(vector(0, -1, 0)))
    
    def reset(self):
        self.pos = vector(0, 0, 0)
        self.velocity = vector(0, 0, 0)
        self.kb_v_ratio = {'w': 0, 's': 0, 'a': 0, 'd': 0}
        self.kb_dir = {'w': 0, 's': 0, 'a': 0, 'd': 0}

class Phys_box(box):
    def __init__(self, _pos: vector, _size: vector, _up: vector,
                 _color=vector(1, 1, 1), _texture=None):
        super().__init__(pos=_pos, size=_size, up=_up, 
                         color=_color, texture=_texture)

class Game(object):
    state = True # execute or not
    
    def __init__(self, game_env: Env):
        self.game_env = game_env
        
        button(text='Reset', pos=game_env.scene.title_anchor, 
               bind=lambda: game_env.reset())
        button(text='Stop',  pos=game_env.scene.title_anchor, 
               bind=lambda: setattr(Game, 'state', False))
        
        self.m_ball = Phys_ball(0, 0, 0, 0.5, textures.earth)
        self.m_ball.state.fly = True
        game_env.items['ball_main'] = self.m_ball
        
        self.valid_key = ['w', 'a', 's','d', ' ', 'p']
        self.kb_state = {'w': False, 'a': False, 's': False,'d': False, ' ': False}
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
        self.m_ball.update(*[key for key, state in self.kb_state.items() if state])
        self.game_env.camera_follow(self.m_ball.pos)
        
        args = [str(x) for x in self.kb_state.items()]
        self.game_env.scene.caption = '{:<12s}, {:<12s}, {:<12s}, {:<12s}, {:<12s}'.format(*args)
    
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