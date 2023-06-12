from wire_relocate import ShadowHandWireRelacateEnv
import numpy as np
import time
env = ShadowHandWireRelacateEnv()
env.render_mode = "human"
it = 0
while it<1000:
    a = np.zeros((26,))

    env.step(a)
    it = it+1
    