import os
import sys

from multiagent import utils
from multiagent.utils import AECEnv, ParallelEnv

# Initializing pygame initializes audio connections through SDL. SDL uses alsa by default on all Linux systems
# SDL connecting to alsa frequently create these giant lists of warnings every time you import an environment using pygame
# DSP is far more benign (and should probably be the default in SDL anyways)

if sys.platform.startswith("linux"):
    os.environ["SDL_AUDIODRIVER"] = "dsp"

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

try:
    import sys

    from farama_notifications import notifications

    if "multiagent" in notifications in notifications["multiagent"]:
        print(notifications["multiagent"], file=sys.stderr)
except Exception:  # nosec
    pass
