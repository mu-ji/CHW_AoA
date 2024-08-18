import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置图像大小和背景颜色
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# 创建动点
dot, = ax.plot([], [], 'ro', markersize=10)

# 定义动点的轨迹
def update(frame):
    x = np.sin(2 * np.pi * frame / 50) * 4 + 5
    y = np.cos(2 * np.pi * frame / 50) * 4 + 5
    dot.set_data(x, y)
    return dot,

# 创建并播放动画
ani = FuncAnimation(fig, update, frames=np.linspace(0, 50, 50), interval=50, blit=True)

# 显示动画
plt.show()