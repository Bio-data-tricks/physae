"""
Functions for parsing HITRAN transition data.
"""
import torch


def parse_csv_transitions(csv_str: str) -> list:
    """
    Parse HITRAN transition data from CSV string.

    Args:
        csv_str: CSV string with transition data (semicolon-separated).

    Returns:
        List of transition dictionaries.
    """
    transitions = []
    for line in csv_str.strip().splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        toks = [t.strip() for t in line.split(";")]
        while len(toks) < 14:
            toks.append('0')
        transitions.append({
            'mid': int(toks[0]),
            'lid': int(float(toks[1])),
            'center': float(toks[2]),
            'amplitude': float(toks[3]),
            'gamma_air': float(toks[4]),
            'gamma_self': float(toks[5]),
            'e0': float(toks[6]),
            'n_air': float(toks[7]),
            'shift_air': float(toks[8]),
            'abundance': float(toks[9]),
            'gDicke': float(toks[10]),
            'nDicke': float(toks[11]),
            'lmf': float(toks[12]),
            'nlmf': float(toks[13]),
        })
    return transitions


def transitions_to_tensors(transitions: list, device) -> list:
    """
    Convert transition dictionaries to PyTorch tensors.

    Args:
        transitions: List of transition dictionaries.
        device: PyTorch device.

    Returns:
        List of tensors for each transition parameter.
    """
    keys = ['amplitude', 'center', 'gamma_air', 'gamma_self', 'n_air',
            'shift_air', 'gDicke', 'nDicke', 'lmf', 'nlmf']
    return [torch.tensor([t[k] for t in transitions], dtype=torch.float32, device=device) for k in keys]
