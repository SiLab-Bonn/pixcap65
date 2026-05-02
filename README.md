# Pixcap 65

Host software for the PixCap65 chip control system based on MIO2 + GPAC hardware.

## Setup

* GPAC 5V
* FPGA-Board 5V
* Keithley 2602A SMU for the clocked charge inputs (Could not be used for biasing as the voltage range is too small.)
* Keithley 2401 SMU for the HV supply; sometimes use to measure all three measurement lines for inter-capacitance estimation.
* Power Supply: Thandar TTI: 

The setup currently works only when using python 2.7, but also higher python versions are possible.
Tests are only performed with
* python 2.7: no obvious bugs or unexpected behaviour
* python 3.13: no bugs within operation, but after closing a measurement class, waiting for some time and reopening **always** a USBTimeoutError occurs.

## Traps
* double import trap: **never** run any of the test scripts directly by using theiry filename but *only* with the `-m` flag of python. In all other cases the class and import hierarchy **will** break.
* 