import numpy as np
import matplotlib.pyplot as plt

def simulate_pi_control(Kp, Ki, setpoint=10.0, disturbance=-2.0, steps=200, dt=0.1):
    time = np.arange(0, steps * dt, dt)
    values = np.zeros(steps)
    value = 0.0 
    integral = 0.0   
    
    for i in range(1, steps):
        error = setpoint - value
        integral += error * dt 
        
        control_output = Kp * error + Ki * integral

        dy_dt = control_output + disturbance
        value += dy_dt * dt
        values[i] = value
        
    return time, values

setpoint = 10.0
Kp = 1.0

Ki_list = [0.0, 0.5, 1.5]

plt.figure(figsize=(10, 6))

for Ki in Ki_list:
    time, values = simulate_pi_control(Kp, Ki, setpoint=setpoint)
    
    if Ki == 0.0:
        label = f'P Control (Kp={Kp}, Ki={Ki})\n - Steady-State Error'
        linestyle = '--'
        color = 'gray'
    else:
        label = f'PI Control (Kp={Kp}, Ki={Ki})'
        linestyle = '-'
        color = None
        
    plt.plot(time, values, label=label, linestyle=linestyle, color=color, linewidth=2)

    if Ki == 1.5:
        peak_idx = np.argmax(values)
        peak_time = time[peak_idx]
        peak_value = values[peak_idx]
        plt.annotate('Overshoot', xy=(peak_time, peak_value), xytext=(peak_time + 2, peak_value + 1.5),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=12, fontweight='bold', color='red')

plt.axhline(y=setpoint, color='r', linestyle=':', label='Setpoint (10.0)', linewidth=2)

plt.title('Controller Response: Eliminating Steady-State Error vs. Introducing Overshoot')
plt.xlabel('Time (s)')
plt.ylabel('Measured Value')
plt.legend(loc='lower right')
plt.grid(True)
plt.ylim(0, 15)
plt.show()