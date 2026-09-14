"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""
import logging

from .analysis_util import *

try:
    # noinspection PyCompatibility
    from collections.abc import Callable, Sized, Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Callable, Sized, Iterable
from warnings import filterwarnings

logger = logging.getLogger(__name__)

cap_counter = 0

filterwarnings("ignore", category=NaturalNameWarning)
filterwarnings("ignore", category=IMinuitWarning)
filterwarnings("ignore", category=np.exceptions.RankWarning, module="jacobi")


if __name__ == '__main__':
    from pixcap65.utility.homogenize_plots import set_params
    try:
        from subprocess import run

        run_result = run(['pdflatex', '--version'], check=True, capture_output=True)
        has_latex = True
        logger.info("The latex compiler to use is: %s", run_result.stdout.decode("utf-8"))
    except (FileNotFoundError, ImportError):
        # proceed as if no latex exists
        logger.exception("Could not verify whether latex exists.")
        has_latex = False
    set_params(latex=has_latex,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors="
                           r"{stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}"
                           r"\sisetup{retain-zero-uncertainty}")
    analyze_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
