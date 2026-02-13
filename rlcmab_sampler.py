import numpy as np

class sampler:
    def __init__(self, roll_number):
        self.roll_number = roll_number
        np.random.seed(roll_number)
        # Mock reward means/stds for each arm (12 arms)
        self.arm_means = np.random.uniform(-5, 10, 12)
        self.arm_stds = np.random.uniform(0.5, 2.0, 12)

    def sample(self, arm_index):
        if not (0 <= arm_index < 12):
            raise ValueError("Arm index must be between 0 and 11")
        # Generate random reward from normal distribution
        return np.random.normal(self.arm_means[arm_index], self.arm_stds[arm_index])
