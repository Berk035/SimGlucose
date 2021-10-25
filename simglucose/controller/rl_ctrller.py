from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class RLController(Controller):
    def __init__(self, P=1, I=0, D=0, target=140):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = 0
        print("It will be implemented!!!")

    def policy(self, observation, reward, done, **kwargs):
        action = 0
        return action

    def reset(self):
        self.integrated_state = 0
        self.prev_state = 0
