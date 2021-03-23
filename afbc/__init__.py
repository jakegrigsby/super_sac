import torch

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

from .agent import AFBCAgent
from . import wrappers
from .main import afbc
from . import nets
from . import replay
