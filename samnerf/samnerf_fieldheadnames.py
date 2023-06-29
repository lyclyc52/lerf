from enum import Enum
# from nerfstudio.field_components.field_heads import FieldHeadNames

class SAMNERFFieldHeadNames(Enum):
    """Possible field outputs"""
    HASHGRID = "hashgrid"
    CLIP = "clip"
    DINO = "dino"
    # SEEM = "seem"
    # SAM = 'sam'