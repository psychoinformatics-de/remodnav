import numpy as np
from remodnav import EyegazeClassifier
from remodnav.tests.test_labeled import load_data as load_anderson

labeled_files = {
    'dots': [
        'TH20_trial1_labelled_{}.mat',
        'TH38_trial1_labelled_{}.mat',
        'TL22_trial17_labelled_{}.mat',
        'TL24_trial17_labelled_{}.mat',
        'UH21_trial17_labelled_{}.mat',
        'UH21_trial1_labelled_{}.mat',
        'UH25_trial1_labelled_{}.mat',
        # following file is missing for RA
        # https://github.com/richardandersson/EyeMovementDetectorEvaluation/issues/1
        #'UH33_trial1_labelled_{}.mat',
        'UL27_trial17_labelled_{}.mat',
        'UL31_trial1_labelled_{}.mat',
        'UL39_trial1_labelled_{}.mat',
    ],
    'images': [
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
    'videos': [
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


for stimtype in ('images', 'dots', 'videos'):
#for stimtype in ('images', 'videos'):
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
            'FIX: %.3f (%.3f) [%i]' % (
                np.mean(fixation_durations),
                np.std(fixation_durations),
                len(fixation_durations)))
        print(
            'SAC: %.3f (%.3f) [%i]' % (
                np.mean(saccade_durations),
                np.std(saccade_durations),
                len(saccade_durations)))
        print(
            'PSO: %.3f (%.3f) [%i]' % (
                np.mean(pso_durations),
                np.std(pso_durations),
                len(pso_durations)))
        print(
            'PURS: %.3f (%.3f) [%i]' % (
                np.mean(purs_durations),
                np.std(purs_durations),
                len(purs_durations)))
