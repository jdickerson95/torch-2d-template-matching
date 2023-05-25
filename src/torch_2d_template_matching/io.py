import mrcfile
import starfile
import torch


def load_mrc_map(
        file_path: str
):
    with mrcfile.open(file_path) as mrc:
        return torch.tensor(mrc.data)

def read_mtf(
        file_path: str
):
    df = starfile.read(file_path)
    frequencies = torch.tensor(df[['_rlnResolutionInversePixel']].to_numpy()).float()
    mtf_amplitudes = torch.tensor(df[['_rlnMtfValue']].to_numpy()).float()
    return frequencies, mtf_amplitudes

