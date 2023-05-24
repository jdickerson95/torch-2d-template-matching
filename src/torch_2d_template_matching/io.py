import mrcfile
import torch

def load_mrc_map(
        file_path: str
):
    with mrcfile.open(file_path) as mrc:
        return torch.tensor(mrc.data)


