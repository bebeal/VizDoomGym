from gym.envs.registration import register

register(
    id="VizdoomGymBasic-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "basic.cfg"}
)

register(
    id="VizdoomGymCorridor-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "deadly_corridor.cfg"}
)

register(
    id="VizdoomGymDefendCenter-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "defend_the_center.cfg"}
)

register(
    id="VizdoomGymDefendLine-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "defend_the_line.cfg"}
)

register(
    id="VizdoomGymHealthGathering-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "health_gathering.cfg"}
)

register(
    id="VizdoomGymMyWayHome-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "my_way_home.cfg"}
)

register(
    id="VizdoomGymPredictPosition-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "predict_position.cfg"}
)

register(
    id="VizdoomGymTakeCover-v0",
    entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "take_cover.cfg"}
)

register(
    id="VizdoomGymDeathmatch-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "deathmatch.cfg"}
)

register(
    id="VizdoomGymHealthGatheringSupreme-v0",
    entry_point="vizdoomenv:DoomEnv",
    kwargs={"scenario_file": "health_gathering_supreme.cfg"}
)
