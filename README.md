# REMoDNaV - Robust Eye Movement Detection for Natural Viewing

[![Build status](https://ci.appveyor.com/api/projects/status/djh7oracomf8qy4s/branch/master?svg=true)](https://ci.appveyor.com/project/mih/remodnav/branch/master) [![codecov.io](https://codecov.io/github/psychoinformatics-de/remodnav/coverage.svg?branch=master)](https://codecov.io/github/psychoinformatics-de/remodnav?branch=master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![GitHub release](https://img.shields.io/github/release/psychoinformatics-de/remodnav.svg)](https://GitHub.com/psychoinformatics-de/remodnav/releases/) [![PyPI version fury.io](https://badge.fury.io/py/remodnav.svg)](https://pypi.python.org/pypi/remodnav/) [![DOI](https://zenodo.org/badge/147316247.svg)](https://zenodo.org/badge/latestdoi/147316247)

REMoDNaV is a velocity based eye movement event detection algorithm that is based on, but
extends the adaptive Nyström & Holmqvist algorithm (Nyström & Holmqvist, 2010).
It is built to be suitable for both static and dynamic stimulation, and is
capable of detecting saccades, post-saccadic oscillations, fixations, and smooth
pursuit events. REMoDNaV is especially suitable for data without a trial structure
and performs robustly on data with temporally varying noise level.


## Support

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/psychoinformatics-de/remodnav

If you have a problem or would like to ask a question about how to use REMoDNaV,
please [submit a question to
NeuroStars.org](https://neurostars.org/new-topic?body=-%20Please%20describe%20the%20problem.%0A-%20What%20steps%20will%20reproduce%20the%20problem%3F%0A-%20What%20version%20of%20REMoDNaV%20are%20you%20using%3F%20On%20what%20operating%20system%20%3F%0A-%20Please%20provide%20any%20additional%20information%20below.%0A-%20Have%20you%20had%20any%20luck%20using%20REMoDNaV%20before%3F%20%28Sometimes%20we%20get%20tired%20of%20reading%20bug%20reports%20all%20day%20and%20a%20lil'%20positive%20end%20note%20does%20wonders%29&tags=remodnav)
with a ``remodnav`` tag.  NeuroStars.org is a platform similar to StackOverflow
but dedicated to neuroinformatics.

Any previous REMoDNaV questions can be found here:
http://neurostars.org/tags/remodnav/


## Installation via pip

Install the latest version of `remodnav` from
[PyPi](https://pypi.org/project/remodnav). It is recommended to use
a dedicated [virtualenv](https://virtualenv.pypa.io):

    # create and enter a new virtual environment (optional)
    virtualenv --python=python3 ~/env/remodnav
    . ~/env/remodnav/bin/activate

    # install from PyPi
    pip install remodnav


## Example usage

**required (positional) arguments:**

REMoDNaV is easiest to use from the command line.
To get REMoDNaV up and running, supply the following required information in a
command line call:
- ``infile``: Data file with eye gaze recordings to process. The first two columns
  in this file must contain x and y coordinates, while each line is a timepoint
  (no header). The file is read with NumPy's ``recfromcsv`` and may be compressed.
  The columns are expected to be seperated by tabulators (``\t``).
- ``outfile``: Output file name. This file will contain information on all detected
  eye movement events in BIDS events.tsv format.
- ``px2deg``: Factor to convert pixel coordinates to visual degrees, i.e. the visual
  angle of a single pixel. Pixels are assumed to be square. This will typically be a
  rather small value.

  Note: you can compute this factor from *screensize*,
  *viewing distance* and *screen resolution* with the following formula:
  ``degrees(atan2(.5 * screen_size, viewing_distance)) / (.5 * screen_resolution)``
- ``sampling rate``: Sampling rate of the data in Hertz. Only data with dense regular
  sampling are supported.

Exemplary command line call:

    remodnav "inputs/raw_eyegaze/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv.gz" \
      "sub-01/sub-01_task-movie_run-1_events.tsv" 0.0185581232561 1000.0

**optional parameters:**

REMoDNaV comes with many configurable parameters. These parameters have sensible default values,
but they can be changed by the user within the command line call.
Further descriptions of these parameters can be found in the corresponding [publication](https://link.springer.com/article/10.3758/s13428-020-01428-x).

| Parameter | Unit   | Description                                                                              |
| -------------------------- | ------ | ---------------------------------------------------------------------------------------- |
| ``--min-blink-duration``| sec |  missing data windows shorter than this duration will not be considered for ``dilate nan``|
| ``--dilate-nan``| sec | duration for which to replace data by missing data markers on either side of a signal-loss window. |
| ``--median-filter-length``| sec | smoothing median-filter size (for initial data chunking only).|
| ``--savgol-length``| sec | size of Savitzky-Golay filter for noise reduction. |
| ``--savgol-polyord``| | polynomial order of Savitzky-Golay filter for noise reduction. |
| ``--max-vel``| deg/sec | maximum velocity threshold, will issue warning if exceeded to inform about potentially inappropriate filter settings. |
| ``--min-saccade_duration``| sec | minimum duration of a saccade event candidate. |
| ``--max-pso_duration``| sec | maximum duration of a post-saccadic oscillation (glissade) candidate. |
| ``--min-fixation_duration``| sec | minimum duration of a fixation event candidate. |
| ``--min-pursuit_duration``| sec | minimum duration of a pursuit event candidate. |
| ``--min-intersaccade_duration``| sec | no saccade detection is performed in windows shorter than twice this value, plus minimum saccade and PSO duration. |
| ``--noise-factor`` |  | adaptive saccade onset threshold velocity is the median absolute deviation of velocities in the window of interest, times this factor (peak velocity threshold is twice the onset velocity); increase for noisy data to reduce false positives (Nyström and Holmqvist, 2010, equivalent: 3.0). |
| ``--velthresh-startvelocity``| deg/sec | start value for adaptive velocity threshold algorithm (Nyström and Holmqvist, 2010), should be larger than any conceivable minimum saccade velocity. |
| ``--max-initial-saccade-freq``| Hz | maximum saccade frequency for initial detection of major saccades, initial data chunking is stopped if this frequency is reached (should be smaller than an expected (natural) saccade frequency in a particular context).|
| ``--saccade-context-window-length``| sec | size of a window centered on any velocity peak for adaptive determination of saccade velocity thresholds (for initial data chunking only). |
| ``--lowpass-cutoff-freq``| Hz | cut-off frequency of a Butterworth low-pass filter applied to determine drift velocities in a pursuit event candidate. |
| ``--pursuit-velthresh``| deg/sec | fixed drift velocity threshold to distinguish periods of pursuit from periods of fixation. |

Thus, to change the default value of any parameter(s), it is sufficient to include the parameter(s) and
the desired value(s) into the command line call:

    remodnav "inputs/raw_eyegaze/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv.gz" \
    "sub-01/sub-01_task-movie_run-1_events.tsv" 0.0185581232561 1000.0 --min-blink-duration 0.05


## Citation

Dar, A. H., Wagner, A. S. & Hanke, M. (2019). [REMoDNaV: Robust Eye Movement Detection for Natural Viewing](https://doi.org/10.1101/619254). *bioRxiv*. DOI: ``10.1101/619254``
*(first two authors contributed equally)*

## License

MIT/Expat


## Contributing

Contributions in the form of issue reports, bug fixes, feature extensions are always
welcome.


## References

Nyström, M., & Holmqvist, K. (2010). [An adaptive algorithm for fixation, saccade, and
glissade detection in eyetracking data](https://doi.org/10.3758/BRM.42.1.188).
Behavior research methods, 42(1), 188-204. DOI: ``10.3758/BRM.42.1.188``
