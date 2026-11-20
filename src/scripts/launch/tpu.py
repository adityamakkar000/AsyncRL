from enum import Enum


class Zone(str, Enum):
    US_CENTRAL1_A = "us-central1-a"
    US_EAST5_A = "us-east5-a"
    US_EAST1_D = "us-east1-d"
    EUROPE_WEST4_A = "europe-west4-a"


class TPUType(str, Enum):
    V5P_8 = "v5p-8"
    V5P_16 = "v5p-16"
    V5P_32 = "v5p-32"
    V5P_64 = "v5p-64"
    V5P_128 = "v5p-128"
    V6E_8 = "v6e-8"
    V6E_32 = "v6e-32"
    V6E_64 = "v6e-64"
    V6E_128 = "v6e-128"

    @staticmethod
    def all_v5() -> list["TPUType"]:
        return [TPUType.V5P_8, TPUType.V5P_16, TPUType.V5P_32, TPUType.V5P_64, TPUType.V5P_128]

    @staticmethod
    def all_v6() -> list["TPUType"]:
        return [TPUType.V6E_8, TPUType.V6E_32, TPUType.V6E_64, TPUType.V6E_128]


class Runtime(str, Enum):
    V2_ALPHA_TPUV5 = "v2-alpha-tpuv5"
    V2_ALPHA_TPUV6E = "v2-alpha-tpuv6e"
