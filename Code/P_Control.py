import numpy as np
import matplotlib.pyplot as plt

def simulate_p_control(Kp, setpoint=10.0, disturbance=-2.0, steps=100, dt=0.1):

    time = np.arange(0, steps * dt, dt)
    values = np.zeros(steps)
    value = 0.0 
    
    for i in range(1, steps):
        error = setpoint - value
        control_output = Kp * error  
        
        dy_dt = control_output + disturbance
        value += dy_dt * dt
        values[i] = value
        
    return time, values

setpoint = 10.0
Kp_list = [0.5, 1.0, 2.0, 5.0] 

plt.figure(figsize=(10, 6))

for Kp in Kp_list:
    time, values = simulate_p_control(Kp, setpoint=setpoint)
    steady_state_error = setpoint - values[-1]
    
    plt.plot(time, values, label=f'Kp = {Kp} (Error: {steady_state_error:.2f})')

plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint (10.0)')

plt.title('P-Controller Response: Effect of Kp on Steady-State Error')
plt.xlabel('Time (s)')
plt.ylabel('Measured Value')
plt.legend(loc='lower right')
plt.grid(True)
plt.ylim(0, 12)
plt.show()