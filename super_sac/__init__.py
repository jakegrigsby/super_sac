import torch

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

from .agent import Agent
from . import wrappers
from .main import super_sac
from . import main
from . import agent
from . import nets
from . import replay
