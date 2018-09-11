# REMoDNaV change log

This is a high level and scarce summary of the changes between releases.
Consult the full history of the [git
repository](http://github.com/psychoinformatics-de/remodnav) for more details.

## X.X (XXX XX, XXXX)

- Ability to distinguish any number of fixation and pursuit events within
  a single inter-saccade-period
- No longer use a maximum amplitude parameter to distinguish pursuits from
  fixations, but use a single velocity threshold instead. The threshold
  is evaluated against a heavily low-pass filtered gaze trajectory, to
  only reflect "smooth" eye movement components (and thereby suppress the
  impact of measurement noise).

## 0.1 (Sep 10, 2018)

- Initial release.
