import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate synthetic 'normal' turbine speed data
time_steps = 100
time = np.linspace(0, 10, time_steps)
normal_speed = 12 + 0.5 * np.sin(2 * np.pi * 0.5 * time) + np.random.normal(0, 0.2, time_steps)

# Introduce faults
fault_speed = np.copy(normal_speed)
fault_speed[40:60] += 2  # speed increase fault

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(time, normal_speed, label='Normal Speed')
plt.plot(time, fault_speed, label='Fault Speed')
plt.title('Simulated Wind Turbine Speed')
plt.xlabel('Time (seconds)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()

from scipy.linalg import inv

def kalman_filter(z, dt=1, process_var=1e-5, measurement_var=0.1):
    x_est = np.zeros(len(z))
    P = 1.0
    Q = process_var
    R = measurement_var
    x_est[0] = z[0]
    
    for k in range(1, len(z)):
        # Prediction update
        x_pred = x_est[k - 1]
        P_pred = P + Q

        # Measurement update
        K = P_pred / (P_pred + R)
        x_est[k] = x_pred + K * (z[k] - x_pred)
        P = (1 - K) * P_pred
    
    return x_est

# Apply Kalman Filter to the fault data
kf_fault_speed = kalman_filter(fault_speed)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(time, fault_speed, label='Measured Speed with Fault')
plt.plot(time, kf_fault_speed, label='Estimated Speed after Kalman Filter')
plt.title('Kalman Filter Application on Fault Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

# New Antecedent/Consequent objects hold universe variables and membership functions
speed = ctrl.Antecedent(np.arange(10, 15, 0.1), 'speed')
fault = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'fault')

# Auto-membership function population is possible with .automf(3, 5, or 7)
speed.automf(3)
fault.automf(3)

# Custom membership functions can be built interactively with a familiar Pythonic API
rule1 = ctrl.Rule(speed['poor'], fault['good'])
rule2 = ctrl.Rule(speed['average'], fault['average'])
rule3 = ctrl.Rule(speed['good'], fault['poor'])

fault_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
faulting = ctrl.ControlSystemSimulation(fault_ctrl)

# Taking input from user
mean_speed = float(input("Enter the mean speed during the fault period: "))
faulting.input['speed'] = mean_speed

# Crunching the numbers
faulting.compute()

print(f"Fault level: {faulting.output['fault']}")

speed.view()
fault.view()
plt.show()
