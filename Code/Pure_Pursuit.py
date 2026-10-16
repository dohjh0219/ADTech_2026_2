import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

cx = np.linspace(0, 50, 500)
cy = np.sin(cx / 3.0) * 5.0

class Vehicle:
    def __init__(self, x, y, yaw, v, L):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L
        self.hx = [x]
        self.hy = [y]

    def update(self, delta, dt):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.L * np.tan(delta) * dt
        self.hx.append(self.x)
        self.hy.append(self.y)

def get_target(v_obj, cx, cy, Ld):
    dists = np.hypot(cx - v_obj.x, cy - v_obj.y)
    idx = np.argmin(dists)
    for i in range(idx, len(cx)):
        if dists[i] > Ld:
            return cx[i], cy[i]
    return cx[-1], cy[-1]

L1 = 1.5
L2 = 8.0
v1 = Vehicle(0, 3, 0, 4.0, 2.5)
v2 = Vehicle(0, 3, 0, 4.0, 2.5)

dt = 0.1
frames = 150
tx1_list = []
ty1_list = []
tx2_list = []
ty2_list = []

for _ in range(frames):
    tx1, ty1 = get_target(v1, cx, cy, L1)
    alpha1 = np.arctan2(ty1 - v1.y, tx1 - v1.x) - v1.yaw
    delta1 = np.arctan2(2.0 * v1.L * np.sin(alpha1), L1)
    v1.update(delta1, dt)
    tx1_list.append(tx1)
    ty1_list.append(ty1)
    
    tx2, ty2 = get_target(v2, cx, cy, L2)
    alpha2 = np.arctan2(ty2 - v2.y, tx2 - v2.x) - v2.yaw
    delta2 = np.arctan2(2.0 * v2.L * np.sin(alpha2), L2)
    v2.update(delta2, dt)
    tx2_list.append(tx2)
    ty2_list.append(ty2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cx, cy, 'k--', label='Reference Path', alpha=0.5)

line1, = ax.plot([], [], 'r-', label=f'Small Ld ({L1})', linewidth=2)
line2, = ax.plot([], [], 'b-', label=f'Large Ld ({L2})', linewidth=2)

car1, = ax.plot([], [], 'ro', markersize=6)
car2, = ax.plot([], [], 'bo', markersize=6)

target1, = ax.plot([], [], 'r*', markersize=8)
target2, = ax.plot([], [], 'b*', markersize=8)

lookahead1, = ax.plot([], [], 'r:', alpha=0.5)
lookahead2, = ax.plot([], [], 'b:', alpha=0.5)

ax.legend(loc='lower right')
ax.set_xlim(-2, 52)
ax.set_ylim(-8, 8)
ax.set_title('Pure Pursuit: Small vs Large Look-ahead Distance')
ax.grid(True)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    car1.set_data([], [])
    car2.set_data([], [])
    target1.set_data([], [])
    target2.set_data([], [])
    lookahead1.set_data([], [])
    lookahead2.set_data([], [])
    return line1, line2, car1, car2, target1, target2, lookahead1, lookahead2

def update(frame):
    line1.set_data(v1.hx[:frame+1], v1.hy[:frame+1])
    line2.set_data(v2.hx[:frame+1], v2.hy[:frame+1])
    
    car1.set_data([v1.hx[frame]], [v1.hy[frame]])
    car2.set_data([v2.hx[frame]], [v2.hy[frame]])
    
    if frame < frames:
        target1.set_data([tx1_list[frame]], [ty1_list[frame]])
        target2.set_data([tx2_list[frame]], [ty2_list[frame]])
        lookahead1.set_data([v1.hx[frame], tx1_list[frame]], [v1.hy[frame], ty1_list[frame]])
        lookahead2.set_data([v2.hx[frame], tx2_list[frame]], [v2.hy[frame], ty2_list[frame]])
        
    return line1, line2, car1, car2, target1, target2, lookahead1, lookahead2

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)

ani.save('pure_pursuit_comparison.gif', writer='pillow', fps=20)
plt.show()