from utils.enums import AgentEnum, EnvironmentEnum, EntropySettingEnum, PlottingSettingEnum

config = {
    "agent": AgentEnum.PPO, # PPO | PPOBETA | SB_PPO | SB_A2C | SB_SAC | RANDOM
    "environment": EnvironmentEnum.ORIGINAL, # ORIGINAL | ADVANCED | RIGHTTURN | MIDDLE | CURVES
    "entropy_setting": EntropySettingEnum.FIXED, # FIXED | SCHEDULED
    "plotting_setting": PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE, # REWARDS | TRACK_DISTANCE | REWARDS_AND_TRACK_DISTANCE
    "zero_brake": False,  # False | True
    "automatic_restart": False, # False | True
    "use_existing_model": "", # name of torch model (e.g: 20230626_162416_model)
    "seed": 42
}
