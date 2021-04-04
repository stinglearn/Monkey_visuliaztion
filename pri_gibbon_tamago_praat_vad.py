# coding:utf-8
import numpy as np
import parselmouth
import datetime
import os
import glob
import sys

def calc_onset_offset(fn,pitch_floor,pitch_ceiling,duration_criterion,interval_criterion):
    snd = parselmouth.Sound(fn)
    pitch = snd.to_pitch(pitch_floor = pitch_floor, pitch_ceiling = pitch_ceiling)
    # pitch = pitch.smooth(bandwidth = 30)
    pitch_values = pitch.selected_array['frequency']
    pitch_values_copy = np.copy(pitch_values) 
    pitch_values[pitch_values > 0] = 1 # binarize
    pitch_value_diff = np.diff(pitch_values) # detect onset or offset frame location

    # init parameter
    dc = int(duration_criterion//pitch.dt) # duraion criterion as frame scale
    ic = int(interval_criterion//pitch.dt) # interval criterion as frame scale

    # calc the initial candidate values, onset, offset, 
    onset_loc_candidate_s = np.where(pitch_value_diff > 0)[0] 
    offset_loc_candidate_s = np.where(pitch_value_diff < 0)[0]
    
    # check length of two loc_candidate
    if len(onset_loc_candidate_s) >= 1:
        if onset_loc_candidate_s[0] > offset_loc_candidate_s[0]:
            onset_loc_candidate_s = np.roll(np.append(onset_loc_candidate_s,0),shift=1)
        if onset_loc_candidate_s[-1] > offset_loc_candidate_s[-1]:
            offset_loc_candidate_s = np.append(offset_loc_candidate_s, len(pitch_values))

    # calc duration and interval
    duration_s = offset_loc_candidate_s - onset_loc_candidate_s
    interval_s = onset_loc_candidate_s - np.roll(offset_loc_candidate_s,shift = 1)
    interval_s_binary = [1 if np.abs(interval) > ic else 0 for interval in interval_s]

    # prepare nparray to log the candidate values
    candidate_s = np.array(
        [
            onset_loc_candidate_s,
            offset_loc_candidate_s, 
            duration_s, 
            interval_s, 
            interval_s_binary,
            ]
        , dtype = int).T

    # interval selection process
    detected_interval_loc_s = np.where(np.abs(candidate_s[:,4]) == 1) # to detect the frames to be remain (socre = 1), otherwise delete the frames (score = 0)

    for i, detected_interval_loc in enumerate(detected_interval_loc_s[0].tolist()):
        if not detected_interval_loc == detected_interval_loc_s[0][-1]:
            candidate_s[detected_interval_loc,1] = candidate_s[detected_interval_loc_s[0][i+1] - 1,1]

    # delete frames of which intervals are under the criterion.
    candidate_int_deleted = candidate_s[candidate_s[:,-1] == 1]

    # determine onset_offset_loc
    candidate_deleted = np.zeros(shape = (candidate_int_deleted.shape[0],3),dtype = int)
    candidate_deleted[:,:2] = candidate_s[candidate_s[:,4] == 1][:,:2]
    candidate_deleted[:,2] = candidate_int_deleted[:,1] - candidate_int_deleted[:,0]
    onset_offset = candidate_deleted[candidate_deleted[:,2] > dc]
    return onset_offset, pitch_values_copy

def main(opt):
    base_path = '/LARGE2/gr10443/animal_song/data/gibbon/PRI/after_sound_absorber_installation'
    if opt == "all":
        fn_s = glob.glob(os.path.join(base_path,'*.wav'))
        fn_s.sort()
    else:
        fn_s = glob.glob(os.path.join(base_path,'20190623-080002_1ch.wav'))
    pitch_floor = 350
    pitch_ceiling = 1500
    duration_criterion = 0.1
    interval_criterion = 0.05
    dt_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_path,dt_now)
    os.makedirs(save_dir,exist_ok=True)
    praat_parameters = ",\n".join([
        "pitch_floor: %i" %pitch_floor,
        "pitch_ceiling: %i" %pitch_ceiling,
        "duration_criterion: %1.3f" %duration_criterion,
        "interval_criterion: %1.3f" %interval_criterion,
    ])

    # save parameters used in Praat (parselmouth)
    with open(os.path.join(save_dir,"praat_param.txt"), "w") as f:
        f.write(praat_parameters)

    for fn in fn_s:
        onset_offset, pitch_values = calc_onset_offset(fn,pitch_floor,pitch_ceiling,duration_criterion,interval_criterion)
        fn_without_ext = os.path.basename(fn).split('.')[0]
        save_npy_fn = os.path.join(save_dir,fn_without_ext)
        np.save(save_npy_fn, onset_offset)
        if opt_2 == "pitch":
            np.save('%s_pitch' %save_npy_fn, pitch_values)


if __name__ == "__main__":
    opt = sys.argv[1]
    opt_2 = sys.argv[2]
    main(opt)