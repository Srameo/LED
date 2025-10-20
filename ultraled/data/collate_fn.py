from torch.utils.data._utils.collate import collate, default_collate_fn_map
from copy import deepcopy
import torch

DEFAUT_COLLATE_FN_MAP = default_collate_fn_map

def starlight_collate_fn():
    collate_fn_map = deepcopy(DEFAUT_COLLATE_FN_MAP)

    def collate_tensor_fn(batch, *, collate_fn_map):
        return torch.cat(batch, 0)

    def collate_list_fn(batch, *, collate_fn_map):
        L = []
        for l in batch:
            L.extend(l)
        return L

    collate_fn_map[torch.Tensor] = collate_tensor_fn
    collate_fn_map[list] = collate_list_fn

    def _collate_fn(batch):
        return collate(batch, collate_fn_map=collate_fn_map)

    return _collate_fn