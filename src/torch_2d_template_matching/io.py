import mrcfile
import starfile
import torch
import mmdf


def load_mrc_map(
        file_path: str
):
    with mrcfile.open(file_path) as mrc:
        return torch.tensor(mrc.data)

def load_model(
    file_path: str,
    pixel_size: float,
) -> torch.Tensor:
    df = mmdf.read(file_path)
    atom_zyx = torch.tensor(df[['z', 'y', 'x']].to_numpy()).float()  # (n_atoms, 3)
    atom_zyx -= torch.mean(atom_zyx, dim=-1, keepdim=True)  # center
    atom_zyx /= pixel_size
    return atom_zyx



def read_mtf(
        file_path: str
):
    df = starfile.read(file_path)
    frequencies = torch.tensor(df[['_rlnResolutionInversePixel']].to_numpy()).float()
    mtf_amplitudes = torch.tensor(df[['_rlnMtfValue']].to_numpy()).float()
    return frequencies, mtf_amplitudes

