
import numpy as np
from .kalman_filter import KalmanFilter
from .degeneracy_analyzer import DegeneracyAnalyzer

class AdaptiveKalmanFilter(KalmanFilter):
    def __init__(self, x_dim, z_dim, degeneracy_threshold=10.0):
        super().__init__(x_dim, z_dim)
        self.analyzer = DegeneracyAnalyzer(threshold=degeneracy_threshold)
        self.is_degenerate = False
        self.condition_number = 0.0

    def update(self, z, H, information_matrix=None):
        """
        Adaptive Update Step.
        If information_matrix (Hessian from scan matching) is provided,
        check for degeneracy and adapt R (measurement noise) accordingly.
        """
        if information_matrix is not None:
            self.is_degenerate, self.condition_number, _ = self.analyzer.analyze(information_matrix)

            if self.is_degenerate:
                # Adaptation Strategy: Inflate Measurement Noise Covariance R
                # This tells the filter to trust the prediction (model) more than the measurement
                # in this degenerate situation.
                print(f"[AKF] Degeneracy detected (Cond No: {self.condition_number:.2f}). Inflating R.")
                
                # Simple scaling adaptation: inflate R by a factor proportional to condition number
                # Or just a fixed large factor.
                scale_factor = min(self.condition_number / 10.0, 100.0) # Cap the scaling
                
                # Temporarily store original R
                R_original = self.R.copy()
                
                # Inflate R
                self.R = self.R * scale_factor
                
                # Perform standard update
                super().update(z, H)
                
                # Restore original R for next step (unless we want persistent degradation)
                self.R = R_original
                return

        # Normal update if not degenerate or info matrix not provided
        super().update(z, H)
