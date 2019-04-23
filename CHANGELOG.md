# REMoDNaV change log

This is a high level and scarce summary of the changes between releases.
Consult the full history of the [git
repository](http://github.com/psychoinformatics-de/remodnav) for more details.

## 0.2 (Apr 23, 2019)

- Ability to distinguish any number of fixation and pursuit events within
  a single inter-saccade-period
- No longer use a maximum amplitude parameter to distinguish pursuits from
  fixations, but use a single velocity threshold instead. The threshold
  is evaluated against a heavily low-pass filtered gaze trajectory, to
  only reflect "smooth" eye movement components (and thereby suppress the
  impact of measurement noise).
- New parameter `noise_factor` that influences the adaptive saccade velocity
  threshold. The saccade onset velocity threshold is the median of all
  sub-threshold velocities plus `noise_factor` times the MAD of these
  velocities. The saccade peak velocity threshold is computed in the same
  fashion, but uses `2x noise_factor`. The default value should work for
  noisy data. Reducing this factor can boost saccade detection sensitivity
  for clean data (e.g. Nyström et al., 2010 use the equivalent of a factor
  of 3.0)

## 0.1 (Sep 10, 2018)

- Initial release.
