import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)