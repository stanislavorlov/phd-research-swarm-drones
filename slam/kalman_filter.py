
import numpy as np

class KalmanFilter:
    def __init__(self, x_dim, z_dim):
        self.x = np.zeros(x_dim) # State vector
        self.P = np.eye(x_dim)   # Covariance matrix
        self.Q = np.eye(x_dim)   # Process noise covariance
        self.R = np.eye(z_dim)   # Measurement noise covariance

    def predict(self, F, u=None, B=None):
        """
        Prediction Step
        x = Fx + Bu
        P = FPF' + Q
        """
        if u is not None and B is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z, H):
        """
        Update Step
        y = z - Hx
        S = HPH' + R
        K = PH'S^-1
        x = x + Ky
        P = (I - KH)P
        """
        y = z - np.dot(H, self.x)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        try:
            K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            print("Singular matrix encountered in S, skipping update")
            return

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])
        self.P = np.dot((I - np.dot(K, H)), self.P)
