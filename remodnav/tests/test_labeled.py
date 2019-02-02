import pytest
import numpy as np
import os.path as op

from .. import clf as CLF


def load_data(category, name, basepath=None):
    from scipy.io import loadmat
    from datalad.api import get

    if basepath is None:
        basepath = op.join(
        'remodnav', 'tests', 'data', 'anderson_etal', 'annotated_data',
        'data used in the article')

    fname = op.join(
        basepath,
        category, name + ('' if name.endswith('.mat') else '.mat'))
    get(fname)
    m = loadmat(fname)
    # viewing distance
    vdist = m['ETdata']['viewDist'][0][0][0][0]
    screen_width = m['ETdata']['screenDim'][0][0][0][0]
    screen_res = m['ETdata']['screenRes'][0][0][0][0]
    px2deg = CLF.deg_per_pixel(screen_width, vdist, screen_res)
    #testing
    #px2deg = 0.001
    sr = float(m['ETdata']['sampFreq'][0][0][0][0])
    data = np.rec.fromarrays([
        m['ETdata']['pos'][0][0][:, 3],
        m['ETdata']['pos'][0][0][:, 4]],
        names=('x', 'y'))
    data[np.logical_and(data['x'] == 0, data['y'] == 0)] = (np.nan, np.nan)

    labels = m['ETdata']['pos'][0][0][:, 5]

    label_remap = {
        1: 'FIXA',
        2: 'SACC',
        3: 'PSO',
        4: 'PURS',
    }
    events = []
    ev_type = None
    ev_start = None
    for i in range(len(labels)):
        s = labels[i]
        if ev_type is None and s in label_remap.keys():
            ev_type = s
            ev_start = i
        elif ev_type is not None and s != ev_type:
            events.append(dict(
                id=len(events),
                label=label_remap.get(ev_type),
                start_time=0.0 if ev_start is None else
                float(ev_start) / sr,
                start_index=0 if ev_start is None else
                ev_start,
                
                end_time=float(i) / sr,
                end_index=i,
            ))
            
            ev_type = s if s in label_remap.keys() else None
            ev_start = i
            
    if ev_type is not None:
        events.append(dict(
            id=len(events),
            label=label_remap.get(ev_type),
            start_time=0.0 if ev_start is None else
            float(ev_start) / sr,
            start_index=0 if ev_start is None else
            ev_start,
            
            end_time=float(i) / sr,
            end_index=i,
            
        ))
    return data, labels, events, px2deg, sr


@pytest.mark.parametrize(
    'name',
    [
        ('dots', 'TH20_trial1_labelled_MN.mat'),
        ('dots', 'TH20_trial1_labelled_RA.mat'),
        ('dots', 'TH38_trial1_labelled_MN.mat'),
        ('dots', 'TH38_trial1_labelled_RA.mat'),
        ('dots', 'TL22_trial17_labelled_MN.mat'),
        ('dots', 'TL22_trial17_labelled_RA.mat'),
        ('dots', 'TL24_trial17_labelled_MN.mat'),
        ('dots', 'TL24_trial17_labelled_RA.mat'),
        ('dots', 'UH21_trial17_labelled_MN.mat'),
        ('dots', 'UH21_trial17_labelled_RA.mat'),
        ('dots', 'UH21_trial1_labelled_MN.mat'),
        ('dots', 'UH21_trial1_labelled_RA.mat'),
        ('dots', 'UH25_trial1_labelled_MN.mat'),
        ('dots', 'UH25_trial1_labelled_RA.mat'),
        ('dots', 'UH33_trial17_labelled_MN.mat'),
        ('dots', 'UH33_trial17_labelled_RA.mat'),
        ('dots', 'UL27_trial17_labelled_MN.mat'),
        ('dots', 'UL27_trial17_labelled_RA.mat'),
        ('dots', 'UL31_trial1_labelled_MN.mat'),
        ('dots', 'UL31_trial1_labelled_RA.mat'),
        ('dots', 'UL39_trial1_labelled_MN.mat'),
        ('dots', 'UL39_trial1_labelled_RA.mat'),
        ('img', 'TH34_img_Europe_labelled_MN.mat'),
        ('img', 'TH34_img_Europe_labelled_RA.mat'),
        ('img', 'TH34_img_vy_labelled_MN.mat'),
        ('img', 'TH34_img_vy_labelled_RA.mat'),
        ('img', 'TL20_img_konijntjes_labelled_MN.mat'),
        ('img', 'TL20_img_konijntjes_labelled_RA.mat'),
        ('img', 'TL28_img_konijntjes_labelled_MN.mat'),
        ('img', 'TL28_img_konijntjes_labelled_RA.mat'),
        ('img', 'UH21_img_Rome_labelled_MN.mat'),
        ('img', 'UH21_img_Rome_labelled_RA.mat'),
        ('img', 'UH27_img_vy_labelled_MN.mat'),
        ('img', 'UH27_img_vy_labelled_RA.mat'),
        ('img', 'UH29_img_Europe_labelled_MN.mat'),
        ('img', 'UH29_img_Europe_labelled_RA.mat'),
        ('img', 'UH33_img_vy_labelled_MN.mat'),
        ('img', 'UH33_img_vy_labelled_RA.mat'),
        ('img', 'UH47_img_Europe_labelled_MN.mat'),
        ('img', 'UH47_img_Europe_labelled_RA.mat'),
        ('img', 'UL23_img_Europe_labelled_MN.mat'),
        ('img', 'UL23_img_Europe_labelled_RA.mat'),
        ('img', 'UL31_img_konijntjes_labelled_MN.mat'),
        ('img', 'UL31_img_konijntjes_labelled_RA.mat'),
        ('img', 'UL39_img_konijntjes_labelled_MN.mat'),
        ('img', 'UL39_img_konijntjes_labelled_RA.mat'),
        ('img', 'UL43_img_Rome_labelled_MN.mat'),
        ('img', 'UL43_img_Rome_labelled_RA.mat'),
        ('img', 'UL47_img_konijntjes_labelled_MN.mat'),
        ('img', 'UL47_img_konijntjes_labelled_RA.mat'),
        ('video', 'TH34_video_BergoDalbana_labelled_MN.mat'),
        ('video', 'TH34_video_BergoDalbana_labelled_RA.mat'),
        ('video', 'TH38_video_dolphin_fov_labelled_MN.mat'),
        ('video', 'TH38_video_dolphin_fov_labelled_RA.mat'),
        ('video', 'TL30_video_triple_jump_labelled_MN.mat'),
        ('video', 'TL30_video_triple_jump_labelled_RA.mat'),
        ('video', 'UH21_video_BergoDalbana_labelled_MN.mat'),
        ('video', 'UH21_video_BergoDalbana_labelled_RA.mat'),
        ('video', 'UH29_video_dolphin_fov_labelled_MN.mat'),
        ('video', 'UH29_video_dolphin_fov_labelled_RA.mat'),
        ('video', 'UH47_video_BergoDalbana_labelled_MN.mat'),
        ('video', 'UH47_video_BergoDalbana_labelled_RA.mat'),
        ('video', 'UL23_video_triple_jump_labelled_MN.mat'),
        ('video', 'UL23_video_triple_jump_labelled_RA.mat'),
        ('video', 'UL27_video_triple_jump_labelled_MN.mat'),
        ('video', 'UL27_video_triple_jump_labelled_RA.mat'),
        ('video', 'UL31_video_triple_jump_labelled_MN.mat'),
        ('video', 'UL31_video_triple_jump_labelled_RA.mat'),
    ])
def test_labeled(name):
    data, target_labels, target_events, px2deg, sr = load_data(name[0], name[1])

    clf = CLF.EyegazeClassifier(
        px2deg=px2deg,
        sampling_rate=sr,
        pursuit_velthresh=5.,
        noise_factor=3.0,
        lowpass_cutoff_freq=10.0,
    )
    p = clf.preproc(
        data,
    )

    events = clf(p)
    for i, ev in enumerate(events[1:]):
        spacing = ev['start_time'] - events[i]['end_time']
        # events are either touching or have a small gap between
        # two saccade-type events (up to min_fixation_duration),
        # or we have missing data
        assert spacing == 0 or \
            (('PS' in events[i]['label'] or 'SAC' in events[i]['label']) and \
             'SAC' in ev['label'] and spacing < 0.04) or \
            np.sum(np.isnan(
                data[int(events[i]['end_time'] * sr):
                     int(ev['start_time'] * sr)]['x']))
        # PSOs are always following a saccade
        if 'PS' in ev['label']:
            assert 'SAC' in events[i]['label']

#    import pylab as pl
#    pl.subplot(211)
#    ut.show_gaze(pp=p, events=events, sampling_rate=sr)
#    pl.subplot(212)
#    ut.show_gaze(pp=p, events=target_events, sampling_rate=sr)
#    pl.show()
