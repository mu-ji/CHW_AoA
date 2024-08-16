import numpy as np

def mean_filter(angle_array):
    return np.mean(angle_array)

def weighted_mean_filter(angle_array):
    angle_std = np.std(angle_array)
    weight_array = [1/np.abs(i-angle_std) for i in angle_array]
    normalized_weight = weight_array / np.linalg.norm(weight_array)

    return np.mean(angle_array*normalized_weight)

def antenna_shape_filter(angle_array):
    if np.mean(angle_array) > 45 and np.mean(angle_array) < 135:
        x = 0.8
        return x/2*angle_array[0] + (0.5-x/2)*angle_array[1] + x/2*angle_array[2] + (0.5-x/2)*angle_array[3]
    else:
        x = 0.2
        return x/2*angle_array[0] + (0.5-x/2)*angle_array[1] + x/2*angle_array[2] + (0.5-x/2)*angle_array[3]
    
class _2d_Kalman_Filter:
    def __init__(self,height,anchor_x,anchor_y,dt):
        self.X = np.array([0, 0]).T     #Esitimation 2*1 (x,y)
        self.Y = np.array([0, 0]).T     #True value  2*1 (x,y)
        self.A = []                     #Decided by the height, anchor_x, and anchor_y
        self.B = []                     #if stable situation, B is zero matrix
        self.U = []                     #if stable situation, U is zero vector
        self.H = []                     #            4*1
        self.Z = []                     #Measurement 2*4
        self.dt = 0                     #Time interval not sure if needed
        self.Q = []                     #Process noise matrix
        self.R = []                     #Measurement noise matrix
        self.P = []
    
    def predict(self):
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.U)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.X = self.X + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
