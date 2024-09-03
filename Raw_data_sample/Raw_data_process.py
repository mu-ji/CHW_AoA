import numpy as np
import matplotlib.pyplot as plt
import AoA_cal_angle

data = np.load('Raw_data_sample/0006_2_(0,0)_data.npz')

print(data['phase_data'].shape)

packet_number = 100
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot([i*0.125 for i in range(576)], data['phase_data'][packet_number])

ax2 = fig.add_subplot(312)
ax2.plot([i*0.125 for i in range(576)], data['mag_data'][packet_number])

ax3 = fig.add_subplot(313)
ax3.plot([i for i in range(200)], data['rssi'])
ax3.plot([packet_number]*2, [np.min(data['rssi']), np.max(data['rssi'])])

plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=4)
reduced_data = pca.fit_transform(np.hstack((data['mag_data'], data['phase_data'])))
#reduced_data = pca.fit_transform(data['phase_data'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:,2])
plt.title("PCA of Time Series Samples")
plt.xlim(-3000,3000)
plt.ylim(-3000,3000)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

#plt.plot([i for i in range(200)], reduced_data[:,0])

packet_number = 0
data = np.load('Raw_data_sample/0006_2_(10,0)_data.npz')

def svd_reconstruct_with_max_value(A):
    U, S, Vt = np.linalg.svd(A)
    S_max = np.zeros_like(A, dtype=float)

    S_max[0, 0] = S[0]  # 最大的奇异值
    if len(S) > 1:
        S_max[1, 1] = S[1]  # 次大的奇异值
    if len(S) > 2:
        S_max[2, 2] = S[2]  # 第三个奇异值
    
    # 重构矩阵
    A_reconstructed = U @ S_max @ Vt
    
    return A_reconstructed

def generate_phase_matrix(switch_data, switch_pattern, reference_ant_data):
    phase_matrix = np.zeros((4,4))
    if switch_pattern == '0006':
        phase_matrix[0][0] = np.mean(AoA_cal_angle.release_jump(switch_data[0,:] - reference_ant_data))
        phase_matrix[0][1] = np.mean(AoA_cal_angle.release_jump(switch_data[1,:] - reference_ant_data))
        phase_matrix[0][2] = np.mean(AoA_cal_angle.release_jump(switch_data[2,:] - reference_ant_data))

        phase_matrix[3][1] = np.mean(AoA_cal_angle.release_jump(switch_data[3,:] - reference_ant_data))
        phase_matrix[3][2] = np.mean(AoA_cal_angle.release_jump(switch_data[4,:] - reference_ant_data))
        phase_matrix[3][3] = np.mean(AoA_cal_angle.release_jump(switch_data[5,:] - reference_ant_data))

        phase_matrix[1][0] = np.mean(AoA_cal_angle.release_jump(switch_data[6,:] - reference_ant_data))
        phase_matrix[2][0] = np.mean(AoA_cal_angle.release_jump(switch_data[7,:] - reference_ant_data))
        phase_matrix[3][0] = np.mean(AoA_cal_angle.release_jump(switch_data[8,:] - reference_ant_data))

        phase_matrix[0][3] = np.mean(AoA_cal_angle.release_jump(switch_data[9,:] - reference_ant_data))
        phase_matrix[1][3] = np.mean(AoA_cal_angle.release_jump(switch_data[10,:] - reference_ant_data))
        phase_matrix[2][3] = np.mean(AoA_cal_angle.release_jump(switch_data[11,:] - reference_ant_data))

        phase_matrix[1][1] = np.mean(AoA_cal_angle.release_jump(switch_data[12,:] - reference_ant_data))
        phase_matrix[1][2] = np.mean(AoA_cal_angle.release_jump(switch_data[13,:] - reference_ant_data))
        phase_matrix[2][1] = np.mean(AoA_cal_angle.release_jump(switch_data[14,:] - reference_ant_data))
        phase_matrix[2][2] = np.mean(AoA_cal_angle.release_jump(switch_data[15,:] - reference_ant_data))

    return phase_matrix

while packet_number < len(data['phase_data']):
    phase_data = data['phase_data'][packet_number]
    mag_data = data['mag_data'][packet_number]
    rssi = data['rssi'][packet_number]
    
    reference_ant_data = np.zeros_like(phase_data)
    antenna_phase_array = np.zeros((4,4,len(phase_data)))
    reference_ant_data[0:64] = phase_data[0:64]
    reference_ant_data,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_ant_data)

    antenna_data_list = []
    i = 72
    while i < 576:
        temp_antenna_data = np.zeros_like(phase_data)
        temp_antenna_data[i:i+8] = phase_data[i:i+8]
        temp_antenna_data = AoA_cal_angle.complete_other_phase_data(temp_antenna_data, reference_slope)
        antenna_data_list.append(temp_antenna_data)
        i = i + 16

    first_switch_data = np.array(antenna_data_list)[:16,:]
    second_switch_data = np.array(antenna_data_list)[16:,:]

    first_phase_matrix = generate_phase_matrix(first_switch_data,'0006', reference_ant_data)
    second_phase_matrix = generate_phase_matrix(second_switch_data,'0006', reference_ant_data)

    diff_x_first = first_phase_matrix[:,1:] - first_phase_matrix[:,0:-1]
    diff_x_second = second_phase_matrix[:,1:] - second_phase_matrix[:,0:-1]

    
    plt.figure(figsize=(6, 4))
    plt.imshow(first_phase_matrix, cmap='coolwarm', interpolation='nearest')

    # 添加颜色条
    plt.colorbar()

    # 添加标题和轴标签
    plt.title('Heatmap of (4, 4) Array')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # 在每个单元格中显示数值
    for (i, j), value in np.ndenumerate(first_phase_matrix):
        #if value < 0:
        #    value = value + 403
        plt.text(j, i, value, ha='center', va='center', color='black')

    # 显示图形
    plt.show()
    

    packet_number += 1


