from enum import Enum

class AgentEnum(str,Enum):
    PPO = "PPO",
    PPOBETA = "PPOBETA"
    RANDOM = "RANDOM",
    SB_PPO = "SB_PPO"
    SB_A2C = "SB_A2C"
    SB_SAC = "SB_SAC"

class EnvironmentEnum(str,Enum):
    ORIGINAL = "ORIGINAL",
    ADVANCED = "ADVANCED"
    RIGHTTURN = "RIGHTTURN"
    MIDDLE = "MIDDLE"
    CURVES = "CURVES"

class EntropySettingEnum(str,Enum):
    FIXED = "FIXED",
    SCHEDULED = "SCHEDULED"

class PlottingSettingEnum(str,Enum):
    REWARDS = "REWARDS",
    TRACK_DISTANCE = "TRACK_DISTANCE"
    REWARDS_AND_TRACK_DISTANCE = "REWARDS_AND_TRACK_DISTANCE"