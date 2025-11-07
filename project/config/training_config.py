"""
Training configurations and hyperparameters.
"""

# This file will contain training-specific configurations
# such as learning rates, batch sizes, number of epochs, etc.

class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(self):
        # These can be set based on the original training setup
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5

        # Add other training configurations as needed
        pass
