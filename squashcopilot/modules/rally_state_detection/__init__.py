"""Rally State Pipeline - A package for detecting rally states in squash videos."""

from .rally_state_detector import RallyStateDetector
from .train_model import RallyStateTrainer
from .model.lstm_model import RallyStateLSTM

__version__ = "0.1.2"
__all__ = ["RallyStateDetector", "RallyStateTrainer", "RallyStateLSTM"]
