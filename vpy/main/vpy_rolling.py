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
class Phys_ball(sphere):
    def __init__(self, _pos, _radius, _texture=None):
        
        super().__init__(pos=_pos, radius=_radius, texture=_texture)
        self._mass = 1
        self._velocity = vector(0, 0, 0)
        self._init_pos = _pos
        
    class state(object):
        # state of ball
        move: bool = False
        fly: bool = False
        
    @ property
    def v(self):
        return self._velocity
    @ v.setter
    def v(self, value):
        self._velocity = value
    
    # ========== update ==========
    def update(self, **kwargs):
        self.state.move = (self.v.mag > 1e-3)
            
        ds = self.v * Env.dt
        self.pos += ds
        self.rotate(angle=ds.cross(vector(0, -1, 0)).mag / self.radius, 
                    axis =ds.cross(vector(0, -1, 0)))
    
    def reset(self):
        self.pos = self._init_pos
        self.v   = vector(0, 0, 0)

class Phys_box(box):
    def __init__(self, _pos: vector, _size: vector, _up: vector,
                 _color=vector(1, 1, 1), _texture=None):
        
        super().__init__(pos=_pos, size=_size, up=_up, 
                         color=_color, texture=_texture)
        
        self._init_pos = _pos

class level_0(object):
    def __init__(self):
        # obstacles init
        
        self.test_plane = Phys_box(vector(0, 0, 0),  vector(20, 0.001, 20),
                                    vector(0, 1, 0), _texture=textures.wood)
        
        # self.gamespawn = Phys_box(vector(0, 0, 0,),  vector(2, 0.001, 2), vector(0, 1, 0),
        #                            _texture=textures.rock)
        
        # self.road_1 = Phys_box(vector(0, 0, 6),  vector(2, 0.001, 10), vector(0, 1, 0), 
        #                        _texture=textures.wood)
        # self.road_2 = Phys_box(vector(6, 0, 10), vector(10, 0.001, 2), vector(0, 1, 0), 
        #                        _texture=textures.wood)
        # self.road_3 = Phys_box(vector(10, 0, 4), vector(2, 0.001, 10), vector(0, 1, 0), 
        #                        _texture=textures.wood)
        pass

class Env(object):
    fps: int   = 100  # frame per second
    dt : float = 0.01 # s
    g  : float = 9.81 # m/s^2
    g_vec: vector = vector(0, -g, 0)
    
    def __init__(self, level_cls: object):
        
        self.scene = canvas(width = 900, height = 600)
        self.items = {}
        self.static_rigid = {}
        self.damping_factor = 0.9
        self.inherit_from_level(level_cls)
        
        self.m_ball = Phys_ball(vector(0, 2, 0), 0.5, textures.earth)
        self.m_ball.state.fly = True
        self.items['main_ball'] = self.m_ball
        
    def inherit_from_level(self, level_cls):
        self.level = level_cls()
        pass
    
    # ========== physics ==========
    
    @ staticmethod
    def gravity(v) -> vector:
        return v + vector(0, -Env.g * Env.dt, 0)
    
    # ========== interaction ==========
    
    def dot_to_plane(self, pos: vector, plane_pos: vector, plane_norm: vector):
        ans = (pos - plane_pos).dot(plane_norm)
        return ans
    
    def static_domain_init(self):
        return
    
    # ========== others ==========
    def perspective(self) -> float:
        return np.arctan2(self.scene.forward.z, self.scene.forward.x)
    
    def camera_follow(self, position: vector):
        self.scene.center = position

    def update(self):
        
        for item in self.items.values():
            item.v = Env.gravity(item.v)
            item.update()
        
        collision_p = self.m_ball.pos - \
            self.level.test_plane.up * self.m_ball.radius
            
        if self.dot_to_plane(collision_p, self.level.test_plane.pos, 
                             self.level.test_plane.up) < 0:
            self.m_ball.pos.y = self.level.test_plane.pos.y + self.m_ball.radius
            self.m_ball._velocity.y *= -1 * self.damping_factor
        
        
class Game(Env):
    state = True # execute or not
    
    levels: dict = {
        '0': level_0,
        }
    
    def __init__(self, level_num: str):
        super().__init__(self.levels[level_num])
        
        button(text='Reset', pos=self.scene.title_anchor, 
               bind=lambda: self.reset())
        button(text='Stop',  pos=self.scene.title_anchor, 
               bind=lambda: setattr(Game, 'state', False))
        
        self.kb_valid_key = ['w', 'a', 's','d', ' ', 'ctrl', 'p']
        self.scene.bind('keydown', self.kb_down_cmd)
        self.scene.bind('keyup',   self.kd_up_cmd)
        self.kb_init()
        
    def control_item(self, item, **kwargs):
        if item in self.items.values():
            if 'velocity' in kwargs:
                item.v += kwargs['velocity']
            if 'acceleration' in kwargs:
                item.v += kwargs['acceleration'] * Env.dt
        
        else: raise ValueError("item not found")
        
    def reset(self, *args):
        self.kb_init()
        
        for item in self.items.values():
            item.reset()
    
    # ========== keyboard ==========
    def kb_init(self):
        # keyboard info
        self.kb_state = {'w': False, 'a': False, 's': False,'d': False, 
                         ' ': False, 'ctrl':False}
        # keyboard controlled movement
        self.kb_dir = {'w': vector(-1, 0, 0), 
                       'a': vector( 0, 0,-1), 
                       's': vector( 1, 0, 0), 
                       'd': vector( 0, 0, 1)}
        self.kb_acc_idx = 0.
        self.kb_acc_vec = vector(0, 0, 0)
        self.kb_v_vec = vector(0, 0, 0)
        self.kb_pre_v_vec = vector(0, 0, 0)
    
    def kb_down_cmd(self, event):
        key = event.key
        
        if key in self.kb_valid_key:
            if key == 'p': # scene info
                print("fov:"        + str(self.scene.fov))
                print("camera.pos:" + str(self.scene.camera.pos))
                print("center:"     + str(self.scene.center))
                print("forward:"    + str(self.scene.forward))
                print("range:"      + str(self.scene.range))
                print("==============================")
            
            elif not self.kb_state[key]:
                self.kb_state[key] = True
    
    def kd_up_cmd(self, event):
        key = event.key
        
        if key in self.kb_valid_key:
            if self.kb_state[key]:
                self.kb_state[key] = False
    
    def kb_dir_update(self):
        # unit vector (x-z plane) of keyboard direction
        # depends on perspective (theta).
        theta = self.perspective()
        
        for key, i in zip(self.kb_dir.keys(), range(4)):
            self.kb_dir[key].x = np.cos(theta - i*np.pi/2)
            self.kb_dir[key].z = np.sin(theta - i*np.pi/2)
    
    def kb_acc_update(self, *event):
        self.kb_dir_update()
        
        wasd = [key for key in self.kb_dir.keys() if self.kb_state[key]]
        
        if wasd:
            # if self.kb_acc_idx < 1:
            #     self.kb_acc_idx += Env.dt
            tmp_dir_vec = vector(0, 0, 0)
            for key in wasd:
                tmp_dir_vec += self.kb_dir[key]
            
            tmp_dir_vec.mag = 10
            self.kb_pre_v_vec = tmp_dir_vec
        
        else:
        #     if self.kb_acc_idx > 0:            
        #         self.kb_acc_idx -= Env.dt
            self.kb_pre_v_vec = vector(0, 0, 0)
            
        if ' ' in event and not self.m_ball.state.fly:
            self.m_ball.v.y = 1
            
        # if ' ' in event:
        #     self.m_ball.pos.y += 0.01
        # if 'ctrl' in event:
        #     self.m_ball.pos.y -= 0.01
        
        self.kb_acc_vec = self.kb_pre_v_vec - self.kb_v_vec
        self.kb_v_vec += self.kb_acc_vec * Env.dt
    
    # ========== total update ==========
    def execute(self):
        kb_events = [key for key, state in self.kb_state.items() if state]
        self.kb_acc_update(*kb_events)
        self.control_item(self.m_ball, acceleration=self.kb_acc_vec)
        
        self.update()
        self.camera_follow(self.m_ball.pos)
        
        args = [str(x) for x in self.kb_state.items()]
        self.scene.caption = \
            '{:<12s}, {:<12s}, {:<12s}, {:<12s}, {:<12s}\n'.format(*args) + \
            'v: {:<5.3f}\n'.format(self.m_ball.v.mag) + \
            'test: {:<5.3f}'.format(self.dot_to_plane(self.m_ball.pos, 
                                                      self.level.test_plane.pos, 
                                                      self.level.test_plane.up))

# ==================================================
# Main
def main():
    global game

    level = '0'
    game = Game(level)
    
    while Game.state:
        rate(Env.fps)
        game.execute()
# 
# ==================================================
# exe
if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.4f s' %(end_time - start_time))
# end
# ==================================================