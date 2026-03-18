# Advanced Thoughts onto modifiying the measurement procedure
## Implementation notes
To be implemented:
* sweep up, followed by sweeping down
* internally handling multiple sweeps of the power supply
* gather high statistics for each frequency before switching to the next bias
* measure the average current multiple times for averaging afterwards and error estimation


## Datastructures
### HDF5 files
Here the columns are ordered by the coordinates of the pixel on the grid, so there is no need for further adjustments into this direction
There is also no need for a ´Event` block.
But we should separate the advancements from investigation and data taking from each other.
Also, it should be possible to either switch the file fast or create a own group/table within one file for each bias voltage investigated or other differences.


### Root Tree
There should be one 'event' per measured pixel

| Branch                                      | Block | Bedeutung                                |
|---------------------------------------------|-------|------------------------------------------|
| eventTime/D                                 | Event | Eventzeit, Sekunden seit 1970            |
| pixel_x/i                                   | Event | x-coordinate of the pixel in the grid    |
| pixel_y/i                                   | Event | y_coordinate of the pixel in the grid    |
| measurements/i                              | Daten | number of measurements used              |
| n_frequencies/i                             | Daten | number of frequencies                    |
| frequencies\[n_frequencies\]/D              | Daten | frequencies measured                     |
| currents\[n_frequencies\]\[measurements\]/D | Daten | Durch die SMU gemessene Ströme           |
| avg_current\[n_frequencies\]/D              | Daten | Mittelwerte der gemessenen Ströme        |
| std_current\[n_frequencies\]/D              | Daten | Standardabweichung der gemessenen Ströme |
|                                             |       |                                          |
|                                             |       |                                          |
|                                             |       |                                          |

And the same after processing the data

| Branch                                      | Block      | Bedeutung                                              |
|---------------------------------------------|------------|--------------------------------------------------------|
| eventTime/D                                 | Event      | Eventzeit, Sekunden seit 1970                          |
| pixel_x/i                                   | Event      | x-coordinate of the pixel on the grid                  |
| pixel_y/i                                   | Event      | y-coordinate of the pixel on the grid                  |
| measurements/i                              | Daten      | number of measurements used                            |
| n_frequencies/i                             | Daten      | number of frequencies investigated                     |
| frequencies\[n_frequencies\]/D              | Daten      | frequencies investigated                               |
| currents\[n_frequencies\]\[measurements\]/D | Daten      | Durch die SMU gemessene Ströme                         |
| avg_current\[n_frequencies\]/D              | Daten      | Mittelwerte der gemessenen Ströme                      |
| std_current\[n_frequencies\]/D              | Daten      | Standardabweichung der gemessenen Ströme               |
| capacitance/D                               | Auswertung | Estimation for the capacitance of the pixel            |
| cap_err/D                                   | Auswertung | Estimation for the capcitance error                    |
| leak/D                                      | Auswertung | Estimation for the pixels leakage current              |
| leak_err/D                                  | Auswertung | Estimation for the error of the pixels leakage current |
| res/D                                       | Auswertung | Estimation for the on-resistance                       |
| res_err/D                                   | Auswertung | Estimation for the error of the on-resistance          |
