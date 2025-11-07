"""
Plotting utilities.
"""
import os
import matplotlib.pyplot as plt


def save_fig(fig, path: str, dpi: int = 150):
    """
    Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure object.
        path: Path where to save the figure.
        dpi: Resolution in dots per inch (default: 150).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
