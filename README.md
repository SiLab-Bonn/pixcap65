# Pixcap 65

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview
Host software for the PixCap65 chip control system based on MIO2 + GPAC hardware.
In addition it provides full implementations of different measurement routines.
[Maybe here is something missing in between.]

To work properly the modular data acquisition (DAQ) framework [basil]() which is developed by [SiLab](https://silab-bonn.github.io/) is required.
Due to restrictions of the bundled firmware for the setup a pinned version of `basil` is required.
This framework is not only used to control the `PixCap65` chip but also the power supplies for performing the measurements.

### Features

* Host Software for controlling the `PixCap65` chip
* Implementation of measurement routines to determine different kinds of capacitances'
* Implementation of analysis routines for the measurement.
* Implementation of plotting routines to visualizes the measurement results.

### Installation
A version is available via PyPi (currently only on the test index):

```bash
pip install -i https://test.pypi.org/simple/ --extra-index https://pypi.org/simple/ --pre pixcap65_DOMI1279
```

> **Note:** The PyPI package may be outdated. Installing from source (below) is recommended to get the latest version.

or for development from source:

```bash
git clone https://github.com/SiLab-Bonn/pixcap65.git
cd pixcap65
pip install -e .
```

Currently the published version of the `basil` dependency onto the index is out of date, but nevertheless useless for usage with `PixCap65` as the firmware is not compatible with the master branch.
A usable branch of `basil` could be found as [Ba-Pixcap-2026/python-compatibility](https://github.com/SiLab-Bonn/basil/tree/Ba-Pixcap-2026/python-compatibility)
When using `uv` as a python project and environment manager the following approach is recommended:
```bash
git clone https://github.com/SiLab-Bonn/pixcap65.git
cd pixcap65
uv sync --all-extras --all-groups
```
(I'm not totally sure whether this will work as expected)

## Setup

### Overview

* FPGA-Board 5V (`MIO2`, uses a `Xilinx Spartan-3` FPGA), handles also the interface of the measurement chip to the PC by a `USB` connection.
* `GPAC` 5V (to adapt the `MIO2` logic level to the logic level of the `PixCap65` chip)
* `PixCap65` PCB (which carries the `PixCap65` chip)
* `Keithley 2602A` SMU for the clocked charge inputs (Could not be used for biasing as the voltage range is too small.)
* `Keithley 2401` SMU for the HV supply; sometimes use to measure all three measurement lines for inter-capacitance
  estimation.
* Power Supply: `Thandar TTI`: used to power the FPGAs and therefore also the `PixCap65` chip.

The setup currently works only when using python 2.7, but also higher python versions are possible.
Tests are only performed with `python3.13`.

* `python 2.7`: no obvious bugs or unexpected behaviour (some of the implementations used in the analysis or the measurement utils are incompatible with `python2.7` by now)
* `python 3.13`: no bugs within operation, but after closing a measurement class, waiting for some time and reopening **always** a USBTimeoutError occurs.

### Usage

The usage of the measurement routines requires a `basil yaml` config file.
Although, it is possible to leave it out and use the bundled default one this is not recommended, as this will make certain assumptions about the connections of the Lab devices and the used serial interfaces.
The default configuration file could be acquired by the command `pixcap65-load-example-configuration`.
The file needs then to be modified for the actual used lab devices, their connection types and the used interfaces.


For convenience also a cli for performing measurements is available and installed together with the package in your python environment.
To get more information about its usage use:
```bash
pixcap65-measurement --help
```

### Traps

* double import trap: **never** run any of the test scripts directly by using their filename but *only* with the `-m`
  flag of python. In all other cases the class and import hierarchy **will** break.

# About the code repository

The package itself is located in [Pixcap65](pixcap65), but there are still some legacy scripts from previous
implementations.
The latter ones were moved to [legacy](legacy).
Some of these scripts are adapted to use the newly implemented APIs of the `pixcap65.py` dut implementation.
The adapeted scripts carry the same name but with `_02` appended to distinguish them from the original ones.
In Addition, also the analysis and plotting scripts as well as some utilities I've used for my bachelor's thesis are provided
as [examples](examples).

The utility code file [homogenize_plots.py](pixcap65/utility/homogenize_plots.py) is taken from code sources provided for the module `physik131: EDV für Physiker`.

# License
If not stated otherwise:

**Host software and older parts of the measurement and analysis routines**
  
    The host software and older parts of the implementations are distributed under the BSD 3-Clause ("BSD New" or "BSD Simplified") License

**New Implementations and analysis framework**
    
    The newer parts of the measurement routines and the analysis/plotting framework is distributed under the Apache-2.0 License


For the licenses of `basil` take a look at [https://basil.readthedocs.io/en/latest/index.html](https://basil.readthedocs.io/en/latest/index.html)


