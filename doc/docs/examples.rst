############
Examples
############
Example of project can be found in `examples folder <https://github.com/SiLab-Bonn/pixcap65/tree/master/examples>`_.

For more use cases check also `tests folder <https://github.com/SiLab-Bonn/basil/tree/master/tests>`_ of the basil
repository onto which this project depends.

There are example scripts originating from the old (legacy) implementations of the measurement scripts.
These scripts include also the analysis of the measured data and their plotting.
On the other hand there are the example implementations used by the analysis and plotting code of a bachelor's thesis
using this project.
These could be divided into two parts, the first one includes the full analysis code but requires the presence of the
corresponding measurement data files in the hdf format.
The second part was used for modelling the computed capacitance's w.r.t. to the geometric properties of the
sensors/pixels.

For example implementations of the measurements itself you may take a look at
:py:mod:`pixcap65.measurements` or the `measurements cli script <https://github.com/SiLab-Bonn/pixcap65/tree/master/pixcap65.measurements.py>`_.

Legacy Implementations
======================
:py:mod:`legacy`

.. automodule:: legacy

:py:mod:`legacy.pixcap65_test`

.. automodule:: legacy.pixcap65_test

:py:mod:`legacy.pixcap65_test_02`

.. automodule:: legacy.pixcap65_test_02

:py:mod:`legacy.pixcap_65_test_load_line`

.. automodule:: legacy.pixcap_65_test_load_line
    :members:

:py:mod:`legacy.pixcap_65_test_load_line_02`

.. automodule:: legacy.pixcap_65_test_load_line_02
    :members:

:py:mod:`legacy.pixcap_65_test_total_cap`

.. automodule:: legacy.pixcap_65_test_total_cap
    :members:

:py:mod:`legacy.pixcap_65_test_total_cap_02`

.. automodule:: legacy.pixcap_65_test_total_cap_02
    :members:

:py:mod:`legacy.pixcap_65_test_inter_cap`

.. automodule:: legacy.pixcap_65_test_inter_cap
    :members:

:py:mod:`legacy.pixcap_65_test_inter_cap_02`

.. automodule:: legacy.pixcap_65_test_inter_cap_02
    :members:

Examples
========

Examples for the analysis procedure
-----------------------------------
:py:mod:`examples.data_constants`

.. automodule:: examples.data_constants
    :members:

:py:mod:`examples.conversion`

.. automodule:: examples.conversion
    :members:

:py:mod:`examples.full_analysis`

.. automodule:: examples.full_analysis
    :members:

:py:mod:`examples.mp_analysis`

.. automodule:: examples.mp_analysis
    :members:

:py:mod:`examples.plotting`

.. automodule:: examples.plotting
    :members:


Examples for the modelling of the capacitances'
-----------------------------------------------
:py:mod:`examples.capacitance_models`
.....................................

.. automodule:: examples.capacitance_models
    :members:

:py:mod:`examples.general_model`

.. automodule:: examples.general_model
    :members:

:py:mod:`examples.inter_capacitance_models`

.. automodule:: examples.inter_capacitance_models
    :members:

:py:mod:`examples.plot_dependencies`

.. automodule:: examples.plot_dependencies
    :members:

:py:mod:`examples.detailed_fits`

.. automodule:: examples.detailed_fits
    :members: