# Advanced Thoughts onto modifying the measurement procedure

## Implementation notes

To be implemented:

* sweep up, followed by sweeping down
* internally handling multiple sweeps of the power supply
* gather high statistics for each frequency before switching to the next bias
* measure the average current multiple times for averaging afterwards and error estimation

## Datastructures

### HDF5 files

Here the columns are ordered by the coordinates of the pixel on the grid, so there is no need for further adjustments
into this direction
There is also no need for a ´Event` block.
But we should separate the advancements from investigation and data taking from each other.
Also, it should be possible to either switch the file fast or create an own group/table within one file for each bias
voltage investigated or other differences.

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
| chi2/D                                      | Auswertung | Chi^2 Estimator at it's minimum for the fit statistics |
| p/D                                         | Auswertung | p-Value of the fit to characterize its g.o.f.          |


| HV            |
|---------------|
| 0V (unbiased) |
| - 40.0 V      |

# Helper for the different keyword arguments which could be encountered

* fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
* use_kafe2: indicates whether kafe2 is used for the fit. (default: False) [boolean]
* apply_contours: indicates whether to determine the contours and try to plot them. (default: False) [boolean]
* plot: indicates whether to plot the data. An output PDF object could be submitted here instead of an explicitly created one. (Default: False) [boolean]
* fit_plot_pdf: PDF object to save the fit figures to.
* test_cap_exclusion: whether to exclude row 0 completely. (default: False) [boolean]
* mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation. [array-like]
* mask_lower: threshold to mask all pixels below this value. [float]
* mask_upper: threshold to mask all pixels above this value. [float]
* distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False) [boolean]
* use_corrected: boolean, indicating whether to use the corrected capacitance for plotting. (data corrected for parasitic capacitances of PixCap65, default: False) [boolean]
* hist_bins: integer, number of bins to use for the histogram. (default: 50) [int]
* plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
  at the same time as matplotlib is not necessarily thread-safe.
* lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
  write/read operations on the same file.
* unit: unit of the capacities presented within the plot. [str]
