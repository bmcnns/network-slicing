import numpy as np
from itertools import combinations

class ActionSpace:
    def __init__(self):
        self.actions = np.zeros((4851, 3), dtype=int)

        # It's assumed that each slice gets at least one unit of bandwidth
        # so we need to allocate 97 units across 3 slices.
        combs = np.array(list(combinations(range(97+2), 2)))

        self.actions[:, 0] = combs[:, 0] + 1
        self.actions[:, 1] = combs[:, 1] - combs[:, 0] - 1 + 1
        self.actions[:, 2] = (99 - 1) - combs[:, 1] + 1
        
    def lookup_action(self, i):
        return self.actions[i]