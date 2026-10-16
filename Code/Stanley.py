import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

cx = np.linspace(0, 50, 500)
cy = np.sin(cx / 3.0) * 5.0
cyaw = np.arctan(np.gradient(cy, cx))

class Vehicle:
    def __init__(self, x, y, yaw, v, L, k):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L
        self.k = k
        self.hx = [x]
        self.hy = [y]

    def update(self, delta, dt):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.L * np.tan(delta) * dt
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))
        self.hx.append(self.x)
        self.hy.append(self.y)

def stanley_control(v_obj, cx, cy, cyaw):
    fx = v_obj.x + v_obj.L * np.cos(v_obj.yaw)
    fy = v_obj.y + v_obj.L * np.sin(v_obj.yaw)
    
    dx = fx - cx
    dy = fy - cy
    dists = np.hypot(dx, dy)
    idx = np.argmin(dists)
    
    front_axle_vec = [-np.cos(v_obj.yaw + np.pi / 2), -np.sin(v_obj.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[idx], dy[idx]], front_axle_vec)
    
    theta_e = cyaw[idx] - v_obj.yaw
    theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))
    
    delta = theta_e + np.arctan2(v_obj.k * error_front_axle, v_obj.v)
    return np.clip(delta, -np.pi / 4, np.pi / 4)

v1 = Vehicle(0, 10, 0, 4.0, 2.5, 0.5)
v2 = Vehicle(0, 10, 0, 4.0, 2.5, 5.0)

dt = 0.1
frames = 150

for _ in range(frames):
    delta1 = stanley_control(v1, cx, cy, cyaw)
    v1.update(delta1, dt)
    
    delta2 = stanley_control(v2, cx, cy, cyaw)
    v2.update(delta2, dt)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cx, cy, 'k--', label='Reference Path', alpha=0.5)

line1, = ax.plot([], [], 'r-', label=f'Small k ({v1.k})', linewidth=2)
line2, = ax.plot([], [], 'b-', label=f'Large k ({v2.k})', linewidth=2)

car1, = ax.plot([], [], 'ro', markersize=6)
car2, = ax.plot([], [], 'bo', markersize=6)

ax.legend(loc='lower right')
ax.set_xlim(-2, 52)
ax.set_ylim(-8, 12)
ax.set_title('Stanley Controller: Effect of Proportional Gain (k)')
ax.grid(True)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    car1.set_data([], [])
    car2.set_data([], [])
    return line1, line2, car1, car2

def update(frame):
    line1.set_data(v1.hx[:frame+1], v1.hy[:frame+1])
    line2.set_data(v2.hx[:frame+1], v2.hy[:frame+1])
    car1.set_data([v1.hx[frame]], [v1.hy[frame]])
    car2.set_data([v2.hx[frame]], [v2.hy[frame]])
    return line1, line2, car1, car2

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)
ani.save('stanley_k_comparison.gif', writer='pillow', fps=20)
plt.show()