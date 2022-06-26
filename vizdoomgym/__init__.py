from gym.envs.registration import register
import vizdoomgym.utils
from vizdoomgym.vizdoomenv import DoomEnv

register(
    id="VizdoomGymBasic-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "basic.cfg"}
)

register(
    id="VizdoomGymCorridor-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "deadly_corridor.cfg"}
)

register(
    id="VizdoomGymDefendCenter-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "defend_the_center.cfg"}
)

register(
    id="VizdoomGymDefendLine-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "defend_the_line.cfg"}
)

register(
    id="VizdoomGymHealthGathering-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "health_gathering.cfg"}
)

register(
    id="VizdoomGymMyWayHome-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "my_way_home.cfg"}
)

register(
    id="VizdoomGymPredictPosition-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "predict_position.cfg"}
)

register(
    id="VizdoomGymTakeCover-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "take_cover.cfg"}
)

register(
    id="VizdoomGymDeathmatch-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "deathmatch.cfg"}
)

register(
    id="VizdoomGymHealthGatheringSupreme-v0",
    entry_point="vizdoomgym.vizdoomenv:DoomEnv",
    kwargs={"scenarios": "health_gathering_supreme.cfg"}
)
