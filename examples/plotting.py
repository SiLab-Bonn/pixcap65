# ----------------------------------------------------------
#  Copyright (c) 2026. SiLab, Institute of Physics, University of Bonn.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------
"""
Handling and demonstration script to generate all the plots used for my bachelor's thesis.
Here only an example on how to use the plotting part of this framework.
"""

import logging
import matplotlib
import time

from pixcap65.plotting_util import mp_plotting_init, error_handler

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    matplotlib.use('PDF')

    logging.basicConfig(level=logging.INFO)
    try:
        from subprocess import run

        run_result = run(['pdflatex', '--version'], check=True, capture_output=True)
        has_latex = True
        logger.info("The latex compiler to use is: %s", run_result.stdout.decode("utf-8"))
    except (FileNotFoundError, ImportError):
        # proceed as if no latex exists
        logger.exception("Could not verify whether latex exists.")
        has_latex = False

    mp_plotting_init('PDF', has_latex)

    # use this attempt to achieve a better performance when generating the plots
    import multiprocessing as mp
    from examples.full_analysis import x1_plotter, \
    x5_plotter, presentation_plotter, bare_sample_plotter_second, x2_plotter_second, x6_plotter, x7_plotter, \
    r13_plotter_second, e1_plotter_second, r1_plotter, x4_plotter

    print(mp.current_process().name)
    print(mp.cpu_count())

    start = time.time()
    with mp.Manager() as manager, mp.Pool(initializer=mp_plotting_init, initargs=("PDF", has_latex,)) as pool:
        tables_lock = manager.RLock()
        presentation_plotter(tables_lock)
        process_handles = [
            bare_sample_plotter_second,
            x1_plotter,
            x2_plotter_second,
            x5_plotter,
            x6_plotter,
            x7_plotter,
            r13_plotter_second,
            e1_plotter_second,
            r1_plotter,
            x4_plotter,
        ]
        processes = [pool.apply_async(handle, (tables_lock,), error_callback=error_handler) for handle in
                     process_handles]

        for p in processes:
            p.wait()
            print("Finished the process; Was it sucessful?", p.successful())
        del tables_lock
    print("Time elapsed: ", time.time() - start)
