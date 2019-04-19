import numpy as np
import os

import logging
lgr = logging.getLogger('studyforrest.utils')


if 'NOISE_SEED' in os.environ:
    seed = int(os.environ['NOISE_SEED'])
else:
    seed = np.random.randint(100000000)
    lgr.warn('RANDOM SEED: %i', seed)
np.random.seed(seed)


def get_noise(size, loc, std):
    noise = np.random.randn(size)
    noise *= std
    noise += loc
    return noise


def get_drift(size, start, dist):
    return np.linspace(start, start + dist, size)


def mk_gaze_sample(
        pre_fix=1000,
        post_fix=1000,
        fix_std=5,
        sacc=20,
        sacc_dist=200,
        pso=30,
        pso_dist=-40,
        start_x=0.0,
        noise_std=2,
        ):
    duration = pre_fix + sacc + pso + post_fix
    samp = np.empty(duration)
    # pre_fix
    t = 0
    pos = start_x
    samp[t:t + pre_fix] = get_noise(pre_fix, pos, fix_std)
    t += pre_fix
    # saccade
    samp[t:t + sacc] = get_drift(sacc, pos, sacc_dist)
    t += sacc
    pos += sacc_dist
    # pso
    samp[t:t + pso] = get_drift(pso, pos, pso_dist)
    t += pso
    pos += pso_dist
    # post fixation
    samp[t:t + post_fix] = get_noise(post_fix, pos, fix_std)
    samp += get_noise(len(samp), 0, noise_std)

    return samp


def expand_samp(samp, y=1000.0):
    n = len(samp)
    return np.core.records.fromarrays([
        samp,
        [y] * n,
        [0.0] * n,
        [0] * n],
        names=['x', 'y', 'pupil', 'frame'])


def samp2file(data, fname):
    np.savetxt(
        fname,
        data.T,
        fmt=['%.1f', '%.1f', '%.1f', '%i'],
        delimiter='\t')


def show_gaze(data=None, pp=None, events=None,
              sampling_rate=1000.0, show_vels=True,
              coord_lim=None, vel_lim=(0, 1000)):
    # using the seaborn-recommended qualitative colormap for colorblind
    colors = {
        'FIXA': '#029e73',
        'PURS': '#de8f05',

        'SACC': '#0173b2',
        'ISAC': '#56b4e9',

        'HPSO': '#cc78bc',
        'IHPS': '#cc78bc',
        'LPSO': '#fbafe4',
        'ILPS': '#fbafe4',

        'PSO': '#cc78bc',
    }

    import pylab as pl
    ax1 = pl.gca()
    if data is not None:
        time_idx = np.linspace(0, len(data) / sampling_rate, len(data))
        ax1.plot(
            time_idx,
            data['x'],
            color='#ca9161', lw=1)
        ax1.plot(
            time_idx,
            data['y'],
            color='#ca9161', lw=1)
    if pp is not None:
        time_idx = np.linspace(0, len(pp) / sampling_rate, len(pp))
        if show_vels:
            vel_color = 'xkcd:gunmetal'
            # instantiate a second axes that shares the same x-axis
            ax2 = ax1.twinx()
            ax2.set_yscale('log')
            ax2.plot(
                time_idx,
                pp['vel'],
                color=vel_color,
                lw=.5,
                alpha=0.8)
            ax2.set_ylabel('Velocity (deg/s)', color=vel_color)
            ax2.tick_params(axis='y', labelcolor=vel_color)
            ax2.set_ylim(vel_lim)
        ax1.plot(
            time_idx,
            pp['x'],
            color='black', lw=1)
        ax1.plot(
            time_idx,
            pp['y'],
            color='black', lw=1)
        if coord_lim is not None:
            ax1.set_ylim(coord_lim)
        #pl.plot(
        #    time_idx,
        #    pp['med_vel'],
        #    color='black')
    if events is not None:
        for ev in events:
            ax1.axvspan(
                ev['start_time'],
                ev['end_time'],
                color=colors[ev['label']],
                alpha=0.3 if ev['label'] in ('FIXA', 'PURS') else 1.0)
            #pl.text(ev['start_time'], 0, '{:.1f}'.format(ev['id']), color='red')
    if data is not None or pp is not None:
        ax1.set_ylabel('Gaze coord. (px)')
        ax1.set_xlabel('Time (seconds)')
        # make figure tight
        duration = \
            float(max(len(data) if data is not None else 0,
                      len(pp) if pp is not None else 0)) / sampling_rate
        ax1.set_xlim(0, duration)


def events2df(events):
    import pandas as pd
    return pd.DataFrame(events)
