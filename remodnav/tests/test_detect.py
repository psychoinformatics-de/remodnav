import os.path as op
import pytest
import numpy as np
from . import utils as ut
from .. import clf as d
import datalad.api as dl


common_args = dict(
    px2deg=0.01,
    sampling_rate=1000.0,
)


def test_no_saccade():
    samp = np.random.randn(1000)
    data = ut.expand_samp(samp, y=0.0)
    clf = d.EyegazeClassifier(**common_args)
    p = clf.preproc(data, savgol_length=0.0, dilate_nan=0)
    # the entire segment is labeled as a fixation
    events = clf(p)
    assert len(events) == 1
    print(events)
    assert events[0]['end_time'] - events[0]['start_time'] == 1.0
    assert events[0]['label'] == 'FIXA'

    # missing data split events
    p[500:510]['x'] = np.nan
    events = clf(p)
    assert len(events) == 2
    assert np.all([e['label'] == 'FIXA' for e in events])

    # size doesn't matter
    p[500:800]['x'] = np.nan
    assert len(clf(p)) == len(events)


def test_one_saccade(tmpdir):
    samp = ut.mk_gaze_sample()

    data = ut.expand_samp(samp, y=0.0)
    clf = d.EyegazeClassifier(**common_args)
    p = clf.preproc(data, dilate_nan=0)
    events = clf(p)
    assert events is not None
    # we find at least the saccade
    assert len(events) > 2
    if len(events) == 4:
        # full set
        evdf = ut.events2df(events)
        assert list(evdf['label']) == ['FIXA', 'ISAC', 'ILPS', 'FIXA'] or \
            list(evdf['label']) == ['FIXA', 'ISAC', 'IHPS', 'FIXA']
        for i in range(0, len(evdf) - 1):
            # complete segmentation
            assert evdf['start_time'][i + 1] == evdf['end_time'][i]
    outfname = tmpdir.mkdir('bids').join("events.tsv").strpath
    d.events2bids_events_tsv(events, outfname)
    fcontent = open(outfname, 'r').readlines()
    assert len(fcontent) == len(events) + 1, 'header plus one event per line'
    assert fcontent[0].startswith('onset\tduration'), 'minimum BIDS headers'
    start_times = [float(line.split('\t')[0]) for line in fcontent[1:]]
    assert start_times == sorted(start_times), \
        'events in file are sorted by start time'


def test_too_long_pso():
    samp = ut.mk_gaze_sample(
        pre_fix=1000,
        post_fix=1000,
        sacc=20,
        sacc_dist=200,
        # just under 30deg/s (max smooth pursuit)
        pso=80,
        pso_dist=100)
    data = ut.expand_samp(samp, y=0.0)
    clf = d.EyegazeClassifier(
        max_initial_saccade_freq=.2,
        **common_args)
    p = clf.preproc(data, dilate_nan=0)
    events = clf(p)
    events = ut.events2df(events)
    # there is no PSO detected
    assert list(events['label']) == ['FIXA', 'SACC', 'FIXA']


@pytest.mark.parametrize('infile', [
    'remodnav/tests/data/studyforrest/sub-32/beh/sub-32_task-movie_run-2_recording-eyegaze_physio.tsv.gz',
    'remodnav/tests/data/studyforrest/sub-09/ses-movie/func/sub-09_ses-movie_task-movie_run-2_recording-eyegaze_physio.tsv.gz',
    'remodnav/tests/data/studyforrest/sub-02/ses-movie/func/sub-02_ses-movie_task-movie_run-5_recording-eyegaze_physio.tsv.gz',
])
def test_real_data(infile):
    dl.get(infile)
    data = np.recfromcsv(
        infile,
        delimiter='\t',
        names=['x', 'y', 'pupil', 'frame'])

    clf = d.EyegazeClassifier(
        #px2deg=0.0185581232561,
        px2deg=0.0266711972026,
        sampling_rate=1000.0)
    p = clf.preproc(data)

    events = clf(
        p[:50000],
        #p,
    )

    evdf = ut.events2df(events)

    labels = list(evdf['label'])
    # find all kinds of events
    for t in ('FIXA', 'PURS', 'SACC', 'LPSO', 'HPSO',
              'ISAC', 'IHPS'):
              # 'ILPS' one file doesn't have any
        assert t in labels
    return

    ut.show_gaze(pp=p[:50000], events=events)
    #ut.show_gaze(pp=p, events=events)
    import pylab as pl
    saccades = evdf[evdf['label'] == 'SACC']
    isaccades = evdf[evdf['label'] == 'ISAC']
    print('#saccades', len(saccades), len(isaccades))
    pl.plot(saccades['amp'], saccades['peak_vel'], '.', alpha=.3)
    pl.plot(isaccades['amp'], isaccades['peak_vel'], '.', alpha=.3)
    pl.show()


@pytest.mark.parametrize('infile', [
    'remodnav/tests/data/studyforrest/sub-32/beh/sub-32_task-movie_run-2_recording-eyegaze_physio.tsv.gz',
])
def test_cmdline(infile, tmpdir):
    import remodnav
    dl.get(infile)
    outfname = tmpdir.mkdir('bids').join("events.tsv").strpath

    remodnav.main(['fake', infile, outfname, '0.0266711972026', '1000'])

    assert op.exists(outfname)
    assert op.exists(outfname[:-4] + '.png')
