import numpy as np
import matplotlib.pyplot as plt

def simulate_pid_control(Kp, Ki, Kd, setpoint, steps, dt):
    time = np.arange(0, steps * dt, dt)
    values = np.zeros(steps)
    value = 0.0
    velocity = 0.0
    integral = 0.0
    prev_error = setpoint - value
    
    for i in range(1, steps):
        error = setpoint - value
        integral += error * dt
        derivative = (error - prev_error) / dt
        
        control_output = Kp * error + Ki * integral + Kd * derivative
        
        acceleration = control_output - 0.5 * velocity
        velocity += acceleration * dt
        value += velocity * dt
        
        values[i] = value
        prev_error = error
        
    return time, values

setpoint = 10.0
steps = 400
dt = 0.1

Kp = 2.0
Ki = 0.5

time, values_pi = simulate_pid_control(Kp, Ki, 0.0, setpoint, steps, dt)
time, values_pid = simulate_pid_control(Kp, Ki, 8.0, setpoint, steps, dt)

plt.figure(figsize=(10, 6))

plt.plot(time, values_pi, label='PI Control (Kd=0.0)', linestyle='--', color='orange', linewidth=2)
plt.plot(time, values_pid, label='PID Control (Kd=8.0)', linestyle='-', color='blue', linewidth=2)

plt.axhline(y=setpoint, color='r', linestyle=':', label='Setpoint (10.0)', linewidth=2)

plt.title('Effect of Derivative (D) Term: Reducing Overshoot vs Increasing Settling Time')
plt.xlabel('Time (s)')
plt.ylabel('Measured Value')
plt.legend(loc='lower right')
plt.grid(True)
plt.ylim(0, 16)
plt.show()