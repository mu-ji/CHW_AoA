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
    def __init__(self,height=1,anchor_x=0,anchor_y=0,dt=0.1):
        self.X = np.zeros((2, 1))                             #Esitimation 2*1 (x,y)

        self.A = np.array([[1, 0], [0, 1]])                     #Decided by the height, anchor_x, and anchor_y if mapping from angle space to cartesian space. For now is from angle space to angle sapce
        self.B = np.array([[0, 0], [0, 0]])                     #if stable situation, B is zero matrix
        self.U = np.array([0, 0]).T                             #if stable situation, U is zero vector
        self.H = np.array([[1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [0, 1],
                           [0, 1],
                           [0, 1],
                           [0, 1]])                                             #            8*2
        self.Z = np.zeros((3, 1))                                             #Measurement 8*1
        self.dt = 0                                             #Time interval not sure if needed
        self.Q = np.diag([0.1, 0.1])                                             #Process noise matrix   2*2
        self.R = np.diag([5,5,5,5,5,5,5,5])                                             #Measurement noise matrix   8*8
        self.P = np.eye(2)                                             #   2*2
    
    def predict(self):
        self.X = np.dot(self.A, self.X)# + np.dot(self.B, self.U)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.X = self.X + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def update_R(self, earlier_measurement_matrix):
        self.R = np.cov(earlier_measurement_matrix)
    