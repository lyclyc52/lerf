from enum import Enum
# from nerfstudio.field_components.field_heads import FieldHeadNames

class LERFFieldHeadNames(Enum):
    """Possible field outputs"""
    HASHGRID = "hashgrid"
    FEATURE = 'feature'
    DINO = "dino"
    CONTRASTIVE = 'contrastive'