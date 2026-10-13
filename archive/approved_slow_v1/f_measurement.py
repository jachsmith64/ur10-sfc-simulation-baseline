"""B's replaceable measurement interface, one shared pre-integration sample."""
import numpy as np

class SharedMeasurement:
    name = 'C_equivalent_shared'

    def update(self, env, remaining, remaining_dot):
        self.y_physical = env.tcp_y()
        self.v_physical = env.tcp_y_velocity()
        self.y_effective = self.y_physical + float(remaining)
        self.v_effective = self.v_physical + float(remaining_dot)
        if not np.isfinite([self.y_effective, self.v_effective]).all():
            raise ValueError('Nonfinite shared equivalent measurement')

    def measure(self, env, y_d, y_d_dot=0.0):
        return self.y_effective-float(y_d), self.v_effective-float(y_d_dot)
