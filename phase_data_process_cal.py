import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

phase_data = np.load('phase_data.npy')

fig = plt.figure()
plt.plot([i*0.125 for i in range(160)],phase_data, marker='*')

i = 8
plt.plot([8*i*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+1)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+2)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+3)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+4)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+5)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+6)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+7)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+8)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+9)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+10)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+11)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+12)*0.125]*2, [-201,201],c = 'b')

plt.xlabel('us')
plt.ylabel('phase')
plt.show()

reference_ant_data = np.zeros_like(phase_data)
ant1_data = np.zeros_like(phase_data)
ant2_data = np.zeros_like(phase_data)

reference_ant_data[0:64] = phase_data[0:64]
ant1_data[72:80] = phase_data[72:80]
ant2_data[88:96] = phase_data[88:96]

def complete_reference_phase_data(reference_ant_data):
    diff = reference_ant_data[1:]-reference_ant_data[:-1]
    for i in range(len(diff)):
        if abs(diff[i]) >= 200:
            diff[i] = 0
    slope = np.mean(diff[diff != 0])
    for i in range(len(reference_ant_data)):
        if reference_ant_data[i] != 0:
            continue
        else:
            next_phase = reference_ant_data[i-1] + slope
            if next_phase > 201:
                next_phase = next_phase - 402
            reference_ant_data[i] = next_phase
    return reference_ant_data,slope/0.125

def find_best_intercept(x, y, slope):
    n = len(x)
    A = np.vstack([x, np.ones(n)]).T
    b = np.sum(y - np.multiply(x,slope)) / n
    return b
    
def complete_other_phase_data(ant_phase_data,reference_slope):
    nonzero_indices = np.nonzero(ant_phase_data)[0]
    min_idx = nonzero_indices[0]
    max_idx = nonzero_indices[-1]

    x = [i*0.125 for i in range(min_idx,max_idx)]
    y = ant_phase_data[min_idx:max_idx]
    for i in range(len(y)-1):
        if y[i]-y[i+1] >= 200:
            y[i+1] = y[i+1] + 402

    intercept = find_best_intercept(x,y,reference_slope)
    first_index = min_idx
    last_index = min_idx + 1
    ant_phase_data[first_index] = reference_slope*first_index*0.125 + intercept
    ant_phase_data[last_index] = reference_slope*last_index*0.125 + intercept
    while first_index != 0 or last_index != (len(ant_phase_data)-1):
        if first_index != 0:
            last_phase = ant_phase_data[first_index] - 0.125*reference_slope
            if last_phase <= -201:
                last_phase = last_phase + 402
            ant_phase_data[first_index-1] = last_phase
            first_index = first_index - 1
        if last_index != (len(ant_phase_data)-1):
            next_phase = ant_phase_data[last_index] + 0.125*reference_slope
            if next_phase > 201:
                next_phase = next_phase - 402
            ant_phase_data[last_index+1] = next_phase
            last_index = last_index + 1
    
    return ant_phase_data

reference_ant_data,reference_slope = complete_reference_phase_data(reference_ant_data)
ant1_data = complete_other_phase_data(ant1_data,reference_slope)
ant2_data = complete_other_phase_data(ant2_data,reference_slope)
plt.figure()
plt.plot([8*i*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+1)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+2)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+3)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+4)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+5)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+6)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+7)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+8)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+9)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+10)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+11)*0.125]*2, [-201,201],c = 'b')
plt.plot([8*(i+12)*0.125]*2, [-201,201],c = 'b')
plt.plot([i*0.125 for i in range(160)],phase_data, marker='*')
plt.plot([i*0.125 for i in range(160)],reference_ant_data,c = 'r',label = 'ANT1_1 (ref)')
plt.plot([i*0.125 for i in range(160)],ant1_data,c = 'g',label = 'ANT1_2')
plt.plot([i*0.125 for i in range(160)],ant2_data,c = 'y',label = 'ANT1_3')
plt.show()

