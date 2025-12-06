"""
Path utilities for RD Galaxy Phase Transition Dataset
======================================================
Provides consistent path resolution across all analysis scripts.

Usage:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
    from path_utils import REPO_ROOT, DATA_DIR, FIGURES_DIR

Or simply:
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent  # For scripts in scripts/analysis or scripts/figures
"""

from pathlib import Path

def get_repo_root(script_file):
    """
    Get repository root from any script location.
    
    Parameters
    ----------
    script_file : str or Path
        __file__ from the calling script
        
    Returns
    -------
    Path
        Path to repository root
        
    Examples
    --------
    >>> REPO_ROOT = get_repo_root(__file__)
    >>> data_file = REPO_ROOT / "data/results/sparc_spirals/tdgl_fits.csv"
    """
    script_path = Path(script_file).resolve()
    
    # Traverse up until we find the repo root (contains 'data' and 'scripts' dirs)
    current = script_path.parent
    while current != current.parent:
        if (current / 'data').exists() and (current / 'scripts').exists():
            return current
        current = current.parent
    
    raise FileNotFoundError(
        f"Could not find repository root from {script_file}. "
        "Expected to find 'data' and 'scripts' directories."
    )

# Common paths (can be imported if this file is in scripts/)
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = REPO_ROOT / 'data'
    RESULTS_DIR = DATA_DIR / 'results'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    FIGURES_DIR = REPO_ROOT / 'figures'
    SCRIPTS_DIR = REPO_ROOT / 'scripts'
except:
    # If imported from unknown location, these won't be defined
    pass
