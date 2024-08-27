import numpy as np
import matplotlib.pyplot as plt

angle_list = [i for i in range(20,170,10)]

raw_angle_list = []
KF_angle_list = []

for angle in angle_list:
    raw_data = np.load('Indoor_experiment_result/raw_angle_{}.npy'.format(angle))
    raw_data = raw_data[50:]
    raw_angle_list.append(raw_data)
    KF_data = np.load('Indoor_experiment_result/KF_angle_{}.npy'.format(angle))
    KF_data = KF_data[50:]
    KF_angle_list.append(KF_data)


fig, ax = plt.subplots(figsize=(8, 6))

# 绘制 Violin Plot
ax.violinplot(raw_angle_list, showmeans=True)
ax.violinplot(KF_angle_list, showmeans=True)
# 设置坐标轴标签和标题
ax.set_xlabel('True Angle')
ax.set_ylabel('Estimated Angle')
ax.set_title('Indoor AoA performance (horizontal direction)')

# 设置 x 轴刻度标签
ax.set_xticks([i for i in range(1,16)])
ax.set_xticklabels([f'{i}' for i in angle_list])

ax.plot([i for i in range(1,16)], [i*10+10 for i in range(1,16)],label='True angle')

rect_1 = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue')
rect_2 = plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='orange')
rect_3 = plt.Rectangle((0, 0), 1, 1, facecolor='g', edgecolor='g')
plt.legend([rect_1, rect_2,rect_3], ['Raw data','Kalman filter','True angle'])
plt.grid()
# 显示图形
plt.show()

