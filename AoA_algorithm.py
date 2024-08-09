import numpy as np
from scipy.linalg import eig
from scipy.linalg import svd


def music_spectrum(iq_data1, iq_data2, iq_data3, iq_data4, num_signals):
    # 将输入信号合并为一个矩阵
    iq_data = np.column_stack((iq_data1, iq_data2, iq_data3, iq_data4))
    
    # 计算协方差矩阵
    R = np.cov(iq_data)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eig(R)
    
    # 选择噪声子空间的特征向量
    noise_eigenvectors = eigenvectors[:, eigenvalues.argsort()[:-num_signals]]
    
    # 构建 MUSIC 谱
    omega = np.linspace(0, 2 * np.pi, 1000)
    music_spectrum = np.zeros_like(omega)
    for i, w in enumerate(omega):
        a = np.exp(-1j * w)
        music_spectrum[i] = 1 / np.linalg.norm(np.dot(noise_eigenvectors.conj().T, a))
    
    # 找到谱峰对应的频率
    peak_indices = music_spectrum.argsort()[-num_signals:]
    frequencies = omega[peak_indices] / (2 * np.pi)
    
    return frequencies, music_spectrum

def esprit(iq_data1, iq_data2, iq_data3, iq_data4, num_signals):

    # 将输入信号合并为一个矩阵
    iq_data = np.column_stack((iq_data1, iq_data2, iq_data3, iq_data4))
    
    # 计算协方差矩阵
    R = np.cov(iq_data)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eig(R)
    
    # 选择信号子空间的特征向量
    signal_eigenvectors = eigenvectors[:, np.argsort(-np.abs(eigenvalues))[:num_signals]]
    
    # 构建 Φ1 和 Φ2 矩阵
    Phi1 = signal_eigenvectors[:-1, :]
    Phi2 = signal_eigenvectors[1:, :]
    
    # 计算 ESPRIT 矩阵
    Psi = np.dot(np.linalg.pinv(Phi1), Phi2)
    
    # 计算特征值
    esprit_eigenvalues = np.linalg.eigvals(Psi)
    
    # 计算频率(以 Hz 为单位)
    frequencies = np.angle(esprit_eigenvalues) / (2 * np.pi)
    
    return frequencies