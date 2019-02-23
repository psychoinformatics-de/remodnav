import numpy as np
import pylab as pl
import seaborn as sns
from remodnav import EyegazeClassifier
from remodnav.tests.test_labeled import load_data as load_anderson
import pdb
#pdb.set_trace() to set breakpoint
import pandas as pd


labeled_files = {
    'dots': [
        'TH20_trial1_labelled_{}.mat',
        'TH38_trial1_labelled_{}.mat',
        'TL22_trial17_labelled_{}.mat',
        'TL24_trial17_labelled_{}.mat',
        'UH21_trial17_labelled_{}.mat',
        'UH21_trial1_labelled_{}.mat',
        'UH25_trial1_labelled_{}.mat',
        'UH33_trial17_labelled_{}.mat',
        'UL27_trial17_labelled_{}.mat',
        'UL31_trial1_labelled_{}.mat',
        'UL39_trial1_labelled_{}.mat',
    ],
    'img': [
        'TH34_img_Europe_labelled_{}.mat',
        'TH34_img_vy_labelled_{}.mat',
        'TL20_img_konijntjes_labelled_{}.mat',
        'TL28_img_konijntjes_labelled_{}.mat',
        'UH21_img_Rome_labelled_{}.mat',
        'UH27_img_vy_labelled_{}.mat',
        'UH29_img_Europe_labelled_{}.mat',
        'UH33_img_vy_labelled_{}.mat',
        'UH47_img_Europe_labelled_{}.mat',
        'UL23_img_Europe_labelled_{}.mat',
        'UL31_img_konijntjes_labelled_{}.mat',
        'UL39_img_konijntjes_labelled_{}.mat',
        'UL43_img_Rome_labelled_{}.mat',
        'UL47_img_konijntjes_labelled_{}.mat',
    ],
    'video': [
        'TH34_video_BergoDalbana_labelled_{}.mat',
        'TH38_video_dolphin_fov_labelled_{}.mat',
        'TL30_video_triple_jump_labelled_{}.mat',
        'UH21_video_BergoDalbana_labelled_{}.mat',
        'UH29_video_dolphin_fov_labelled_{}.mat',
        'UH47_video_BergoDalbana_labelled_{}.mat',
        'UL23_video_triple_jump_labelled_{}.mat',
        'UL27_video_triple_jump_labelled_{}.mat',
        'UL31_video_triple_jump_labelled_{}.mat',
    ],
}


def get_durations(events, evcodes):
    events = [e for e in events if e['label'] in evcodes]
    # TODO minus one sample at the end?
    durations = [e['end_time'] - e['start_time'] for e in events]
    return durations

def print_duration_stats():
    for stimtype in ('img', 'dots', 'video'):
    #for stimtype in ('img', 'video'):
        for coder in ('MN', 'RA'):
            print(stimtype, coder)
            fixation_durations = []
            saccade_durations = []
            pso_durations = []
            purs_durations = []
            for fname in labeled_files[stimtype]:
                data, target_labels, target_events, px2deg, sr = load_anderson(
                    stimtype, fname.format(coder))
                fixation_durations.extend(get_durations(
                    target_events, ['FIXA']))
                saccade_durations.extend(get_durations(
                    target_events, ['SACC']))
                pso_durations.extend(get_durations(
                    target_events, ['PSO']))
                purs_durations.extend(get_durations(
                    target_events, ['PURS']))
            print(
                'FIX: %i (%i) [%i]' % (
                    np.mean(fixation_durations) * 1000,
                    np.std(fixation_durations) * 1000,
                    len(fixation_durations)))
            print(
                'SAC: %i (%i) [%i]' % (
                    np.mean(saccade_durations) * 1000,
                    np.std(saccade_durations) * 1000,
                    len(saccade_durations)))
            print(
                'PSO: %i (%i) [%i]' % (
                    np.mean(pso_durations) * 1000,
                    np.std(pso_durations) * 1000,
                    len(pso_durations)))
            print(
                'PURS: %i (%i) [%i]' % (
                    np.mean(purs_durations) * 1000,
                    np.std(purs_durations) * 1000,
                    len(purs_durations)))


def remodnav_on_anderson_mainseq(superimp = "trials"):
    """ by default will make main sequences for each trial/file.
    superimp = "stimulus" for superimposed main sequences of each stimulus 
    type"""
    for stimtype in ('img', 'dots', 'video'):
    #for stimtype in ('img', 'video'):
        if superimp == "stimulus":
            pl.figure(figsize=(6,4))
            
        coder  = 'MN'
        print(stimtype, coder)
        fixation_durations = []
        saccade_durations = []
        pso_durations = []
        purs_durations = []
        for fname in labeled_files[stimtype]:
            data, target_labels, target_events, px2deg, sr = load_anderson(
                stimtype, fname.format(coder))

            clf = EyegazeClassifier(
                px2deg=px2deg,
                sampling_rate=sr,
                pursuit_velthresh=5.,
                noise_factor=3.0,
                lowpass_cutoff_freq=10.0,
            )
            p = clf.preproc(data)
            events = clf(p)
            events = pd.DataFrame(events)
            saccades = events[events['label'] == 'SACC']
            isaccades = events[events['label'] == 'ISAC']
            hvpso = events[(events['label'] == 'HPSO') | (events['label'] == 'IHPS')]
            lvpso = events[(events['label'] == 'LPSO') | (events['label'] == 'ILPS')]

            if superimp == "trials": 
                pl.figure(figsize=(6,4))
            for ev, sym, color, label in (
                    (saccades, '.', 'xkcd:green grey', 'Segment defining saccade'),
                    (isaccades, '.', 'xkcd:dark olive', 'Saccades'),
                    (hvpso, '+', 'xkcd:pinkish', 'High velocity PSOs'),
                    (lvpso, '+', 'xkcd:wine', 'PSOs'))[::-1]:
                pl.loglog(ev['amp'], ev['peak_vel'], sym, color=color,
                        alpha=1, lw=1, label=label)

            pl.ylim((10.0, 1000)) #previously args.max_vel, put this back in
            pl.xlim((0.01, 40.0))
            pl.legend(loc=4)
            pl.ylabel('peak velocities (deg/s)')
            pl.xlabel('amplitude (deg)')
            if superimp == "trials":
                pl.savefig(
                    '{}_{}_remodnav_on_testdata_mainseq.svg'.format(stimtype,fname[0:15]),bbox_inches='tight', format='svg')
                    
        if superimp == "stimulus":
            pl.savefig(
                '{}_remodnav_on_testdata_superimp_mainseq.svg'.format(stimtype,fname[0:15]),bbox_inches='tight', format='svg')
        pl.close('all')   

def preproc_on_anderson_mainseq():
    #for sequentially making main sequences of all the available files
    for stimtype in ('img', 'dots', 'video'):
    #for stimtype in ('img', 'video'):
        for coder in ('MN', 'RA'):
            print(stimtype, coder)
            fixation_durations = []
            saccade_durations = []
            pso_durations = []
            purs_durations = []
            for fname in labeled_files[stimtype]:
                data, target_labels, target_events, px2deg, sr = load_anderson( 
                    stimtype, fname.format(coder))

                clf = EyegazeClassifier(
                    px2deg=px2deg,
                    sampling_rate=sr,
                    pursuit_velthresh=5.,
                    noise_factor=3.0,
                    lowpass_cutoff_freq=10.0,
                )
                pproc = clf.preproc(data)
                pproc_df = pd.DataFrame(pproc) 
                target_events_df = pd.DataFrame(target_events)
                
                saccade_events = target_events_df[target_events_df.label == "SACC"]
                peak_vels = []
                amp       = []
                for row in target_events_df.itertuples():
                    peak_vels.append(pproc_df.vel.loc[row.start_index:row.end_index].max())
                    amp.append ((((pproc_df.x.loc[row.start_index] - pproc_df.x.loc[row.end_index]) ** 2 + \
                    (pproc_df.y.loc[row.start_index] - pproc_df.y.loc[row.end_index]) ** 2) ** 0.5) * px2deg)
                
                peaks_amps_df = pd.DataFrame({'peak_vels':peak_vels,'amp':amp})
                target_events_df= pd.concat([target_events_df, peaks_amps_df], axis=1)
                
                saccades = target_events_df[target_events_df['label'] == 'SACC']
                pso = target_events_df[target_events_df['label'] == 'PSO']

                pl.figure(figsize=(6,4))
                for ev, sym, color, label in (
                        (saccades, '.', 'black', 'saccades'),
                        (pso, '+', 'xkcd:burnt sienna', 'PSOs'))[::-1]:
                    pl.loglog(ev['amp'], ev['peak_vels'], sym, color=color,
                            alpha=.2, lw=1, label=label)

                pl.ylim((10.0, 1000)) #previously args.max_vel, put this back in
                pl.xlim((0.01, 40.0))
                pl.legend(loc=4)
                pl.ylabel('peak velocities (deg/s)')
                pl.xlabel('amplitude (deg)')
                pl.tick_params(which='both',direction = 'in')
                pl.savefig(
                    '{}_{}_{}_mainseq_preproc_on_anderson.svg'.format(stimtype, coder,fname[0:15]),bbox_inches='tight', format='svg')
                
                print(len(peak_vels))
                print(len(amp))
                
def preproc_on_anderson_mainseq_superimp(superimp = "coders"):
    """ by default will make main sequences for each coder for each file
    "stimulus" for superimposed main sequences of each stimulus type"""
    #for making main sequences with Human coders superimposed on one another
    
    for stimtype in ('img', 'dots', 'video'):
    #for stimtype in ('img', 'video'):
        if superimp == "stimulus":
            pl.figure(figsize=(6,4))
            
        for coder in ('MN', 'RA'):
            print(stimtype, coder)
            fixation_durations = []
            saccade_durations = []
            pso_durations = []
            purs_durations = []
            for fname in labeled_files[stimtype]:
                data, target_labels, target_events, px2deg, sr = load_anderson( #change to load_anderson
                    stimtype, fname.format(coder))

                clf = EyegazeClassifier(
                    px2deg=px2deg,
                    sampling_rate=sr,
                    pursuit_velthresh=5.,
                    noise_factor=3.0,
                    lowpass_cutoff_freq=10.0,
                )
                pproc = clf.preproc(data)
                pproc_df = pd.DataFrame(pproc) 
                target_events_df = pd.DataFrame(target_events)
                
                saccade_events = target_events_df[target_events_df.label == "SACC"]
                peak_vels = []
                amp       = []
                for row in target_events_df.itertuples():
                    peak_vels.append(pproc_df.vel.loc[row.start_index:row.end_index].max())
                    amp.append ((((pproc_df.x.loc[row.start_index] - pproc_df.x.loc[row.end_index]) ** 2 + \
                    (pproc_df.y.loc[row.start_index] - pproc_df.y.loc[row.end_index]) ** 2) ** 0.5) * px2deg)
                
                peaks_amps_df = pd.DataFrame({'peak_vels':peak_vels,'amp':amp})
                target_events_df= pd.concat([target_events_df, peaks_amps_df], axis=1)
                
                saccades = target_events_df[target_events_df['label'] == 'SACC']
                pso = target_events_df[target_events_df['label'] == 'PSO']
                
                
                
                if coder == 'MN':
                    if superimp == "coders":
                        pl.figure(figsize=(6,4))
                    for ev, sym, color, label in (
                            (saccades, '.', 'red', 'saccades'),
                            (pso, '+', 'red', 'PSOs'))[::-1]:
                        pl.loglog(ev['amp'], ev['peak_vels'], sym, color=color,
                                alpha=1, lw=1, label=label)
                                
                    pl.ylim((10.0, 1000)) #TODO previously args.max_vel, put this back in
                    pl.xlim((0.01, 40.0))
                    pl.legend(loc=4)
                    pl.ylabel('peak velocities (deg/s)')
                    pl.xlabel('amplitude (deg)')
                    pl.tick_params(which='both' ,direction = 'in')
                    
                    superimp_figure_index = 1
                    
                elif coder == 'RA':
                    if superimp == "coders":
                        pl.figure(superimp_figure_index)
                    for ev, sym, color, label in (
                            (saccades, '.', 'blue', 'saccades'),
                            (pso, '+', 'blue', 'PSOs'))[::-1]:
                        pl.loglog(ev['amp'], ev['peak_vels'], sym, color=color,
                                alpha=1, lw=1, label=label)
                    if superimp == "coders":           
                        pl.savefig(
                            '{}_{}_{}_mainseq_preproc_on_anderson_superimposed.svg'.format(stimtype, coder,fname[0:15]),bbox_inches='tight', format='svg')
                
                    superimp_figure_index += 1

                print(len(peak_vels))
                print(len(amp))
        if superimp == "stimulus":
            pl.savefig(
                '{}_mainseq_preproc_on_anderson_superimposed.svg'.format(stimtype ),bbox_inches='tight', format='svg')
                     
        # Closing set of plots made for each stimulus type
        pl.close('all')
                

def confusion(refcoder, coder):
    conditions = ['FIX', 'SAC', 'PSO', 'PUR']
    #conditions = ['FIX', 'SAC', 'PSO']
    label_map = {
        'FIXA': 'FIX',
        'FIX': 'FIX',
        'SACC': 'SAC',
        'ISAC': 'SAC',
        'HPSO': 'PSO',
        'IHPS': 'PSO',
        'LPSO': 'PSO',
        'ILPS': 'PSO',
        'PURS': 'PUR',
    }
    anderson_remap = {
        'FIX': 1,
        'SAC': 2,
        'PSO': 3,
        'PUR': 4,
    }
    plotter = 1
    pl.suptitle('Jaccard index for movement class labeling {} vs. {}'.format(
        refcoder, coder))
    for stimtype in ('img', 'dots', 'video'):
        conf = np.zeros((len(conditions), len(conditions)), dtype=float)
        jinter = np.zeros((len(conditions), len(conditions)), dtype=float)
        junion = np.zeros((len(conditions), len(conditions)), dtype=float)
        for fname in labeled_files[stimtype]:
            labels = []
            data = None
            px2deg = None
            sr = None
            for src in (refcoder, coder):
                if src in ('RA', 'MN'):
                    data, target_labels, target_events, px2deg, sr = load_anderson(
                        stimtype, fname.format(src))
                    labels.append(target_labels.astype(int))
                else:
                    clf = EyegazeClassifier(
                        px2deg=px2deg,
                        sampling_rate=sr,
                        pursuit_velthresh=5.,
                        noise_factor=3.0,
                        lowpass_cutoff_freq=10.0,
                        min_fixation_duration=0.055,
                    )
                    p = clf.preproc(data)
                    events = clf(p)

                    # convert event list into anderson-style label array
                    l = np.zeros(labels[0].shape, labels[0].dtype)
                    for ev in events:
                        l[int(ev['start_time'] * sr):int((ev['end_time'])* sr)] = \
                            anderson_remap[label_map[ev['label']]]
                    labels.append(l)

            nlabels = [len(l) for l in labels]
            if len(np.unique(nlabels)) > 1:
                print(
                    "#\n# INCONSISTENCY Found label length mismatch between "
                    "coders ({}, {}) for: {}\n#\n".format(
                        refcoder, coder, fname))
                print('Truncate labels to shorter sample: {}'.format(
                    nlabels))
                order_idx = np.array(nlabels).argsort()
                labels[order_idx[1]] = \
                    labels[order_idx[1]][:len(labels[order_idx[0]])]

            for c1, c1label in enumerate(conditions):
                for c2, c2label in enumerate(conditions):
                    intersec = np.sum(np.logical_and(
                        labels[0] == anderson_remap[c1label],
                        labels[1] == anderson_remap[c2label]))
                    union = np.sum(np.logical_or(
                        labels[0] == anderson_remap[c1label],
                        labels[1] == anderson_remap[c2label]))
                    jinter[c1, c2] += intersec
                    junion[c1, c2] += union
                    #if c1 == c2:
                    #    continue
                    conf[c1, c2] += np.sum(np.logical_and(
                        labels[0] == anderson_remap[c1label],
                        labels[1] == anderson_remap[c2label]))

        nsamples = np.sum(conf)
        nsamples_nopurs = np.sum(conf[:3, :3])
        # zero out diagonal for bandwidth
        conf *= ((np.eye(len(conditions)) - 1) * -1)
        pl.subplot(1, 3, plotter)
        sns.heatmap(
            #(conf / nsamples) * 100,
            jinter / junion,
            square=True,
            annot=True,
            xticklabels=conditions,
            yticklabels=conditions,
            vmin=0.0,
            vmax=1.0,
        )
        pl.xlabel('{} labeling'.format(refcoder))
        pl.ylabel('{} labeling'.format(coder))
        pl.title('"{}" (glob. misclf-rate): {:.1f}% (w/o pursuit: {:.1f}%)'.format(
            stimtype,
            (np.sum(conf) / nsamples) * 100,
            (np.sum(conf[:3, :3]) / nsamples_nopurs) * 100))
        plotter += 1
        msclf_refcoder = dict(zip(conditions, conf.sum(axis=1)/conf.sum() * 100))
        msclf_coder = dict(zip(conditions, conf.sum(axis=0)/conf.sum() * 100))
        print('### {}'.format(stimtype))
        print('Comparison | MCLF | MCLFw/oP | Method | Fix | Sacc | PSO | SP')
        print('--- | --- | --- | --- | --- | --- | --- | ---')
        print('{} v {} | {:.1f} | {:.1f} | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f}'.format(
            refcoder,
            coder,
            (np.sum(conf) / nsamples) * 100,
            (np.sum(conf[:3, :3]) / nsamples_nopurs) * 100,
            refcoder,
            msclf_refcoder['FIX'],
            msclf_refcoder['SAC'],
            msclf_refcoder['PSO'],
            msclf_refcoder['PUR'],
        ))
        print('-- | --  | -- | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f}'.format(
            coder,
            msclf_coder['FIX'],
            msclf_coder['SAC'],
            msclf_coder['PSO'],
            msclf_coder['PUR'],
        ))



#confusion('MN', 'RA')
#pl.show()
#confusion('MN', 'ALGO')
#pl.show()
#confusion('RA', 'ALGO')
#pl.show()
#print_duration_stats()
#preproc_on_anderson_mainseq()
#remodnav_on_anderson_mainseq()
#remodnav_on_anderson_mainseq("stimulus")

#For the paper:
#preproc_on_anderson_mainseq_superimp()
#preproc_on_anderson_mainseq_superimp("stimulus")
#remodnav_on_anderson_mainseq()
#remodnav_on_anderson_mainseq("stimulus")
