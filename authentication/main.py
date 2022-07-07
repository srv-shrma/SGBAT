import os
import copy
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mne
from torch.utils.data import Dataset, ConcatDataset
import pickle
import keras


def load_sleep_physionet_raw(raw_fname, annot_fname, load_eeg_only=True, 
                             crop_wake_mins=30):
    """
    Parameters
    ----------
    raw_fname : str
        Path to the .edf file containing the raw data.
    annot_fname : str
        Path to the annotation file.
    load_eeg_only : bool
        If True, only keep EEG channels and discard other modalities 
        (speeds up loading).
    crop_wake_mins : float
        Number of minutes of wake events before and after sleep events.
    
    Returns
    -------
    mne.io.Raw :
        Raw object containing the EEG and annotations.        
    """
    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}
    exclude = mapping.keys() if load_eeg_only else ()
    
    raw = mne.io.read_raw_edf(raw_fname, exclude=exclude)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots, emit_warning=False)
    if not load_eeg_only:
        raw.set_channel_types(mapping)
    
    if crop_wake_mins > 0:  # Cut start and end Wake periods
        # Find first and last sleep stages
        mask = [x[-1] in ['1', '2', '3', '4', 'R'] 
                for x in annots.description]
        sleep_event_inds = np.where(mask)[0]

        # Crop raw
        tmin = annots[int(sleep_event_inds[0])]['onset'] - \
               crop_wake_mins * 60
        tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
               crop_wake_mins * 60
        raw.crop(tmin=tmin, tmax=tmax)
    
    # Rename EEG channels
    ch_names = {i: i.replace('EEG ', '') 
                for i in raw.ch_names if 'EEG' in i}
    mne.rename_channels(raw.info, ch_names)
    
    # Save subject and recording information in raw.info
    basename = os.path.basename(raw_fname)
    subj_nb, rec_nb = int(basename[3:5]), int(basename[5])
    raw.info['subject_info'] = {'id': subj_nb, 'rec_id': rec_nb}
   
    return raw

def extract_epochs(raw, chunk_duration=30.):
    """Extract non-overlapping epochs from raw data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to be windowed.
    chunk_duration : float
        Length of a window.
    
    Returns
    -------
    np.ndarray
        Epoched data, of shape (n_epochs, n_channels, n_times).
    np.ndarray
        Event identifiers for each epoch, shape (n_epochs,).
    """
    annotation_desc_2_event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3': 4,
        'Sleep stage 4': 4,
        'Sleep stage R': 5}

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, 
        chunk_duration=chunk_duration)

    # create a new event_id that unifies stages 3 and 4
    event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3/4': 4,
        'Sleep stage R': 5}

    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    epochs = mne.Epochs(raw=raw, events=events, picks=picks, preload=True,
                        event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    
    return epochs.get_data(), epochs.events[:, 2] - 1

class EpochsDataset(Dataset):
    """Class to expose an MNE Epochs object as PyTorch dataset.
    
    Parameters
    ----------
    epochs_data : np.ndarray
        The epochs data, shape (n_epochs, n_channels, n_times).
    epochs_labels : np.ndarray
        The epochs labels, shape (n_epochs,)
    subj_nb: None | int
        Subject number.
    rec_nb: None | int
        Recording number.
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """
    def __init__(self, epochs_data, epochs_labels, subj_nb=None, 
                 rec_nb=None, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.subj_nb = subj_nb
        self.rec_nb = rec_nb
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X[None, ...])
        return X, y
    
def scale(X):
    """Standard scaling of data along the last dimention.
    
    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        The input signals.
        
    Returns
    -------
    X_t : array, shape (n_channels, n_times)
        The scaled signals.
    """
    X -= np.mean(X, axis=1, keepdims=True)
    return X / np.std(X, axis=1, keepdims=True)

def extract_stage(stage,all_datasets,dataset):
  data_1 = [[] for i in range(0,1)]
  data_2 = [[] for i in range(0,1)]
  count = 0
  for i in range(0,1):
    limit = len(all_datasets[i])
    for j in range(0,limit):
      data = dataset[j]
      x,y = data # x contains readings of both electrodes and y contains sleep stage

      if(y == stage):
        a = x[0][0].numpy() #numpy array having reading of Electrode 1
        b = x[0][1].numpy() #numpy array having reading of Electrode 2
        data_1[i] += a.tolist()
        data_2[i] += b.tolist()
        
        count+=1

  return data_1,data_2,count

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band."""

    """
    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """

    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def Process(all_datasets,dataset):
    X = []
    y = []
    sf = 100.0 #Sampling Frequecy 100.0Hz
    window = 1.0 #Window Size
    bands=[[0.5,4.0],[4.0,8.0],[8.0,12.0],[12.0,30.0]]

    for i in range(0,5):
        signal1, signal2,count = extract_stage(i,all_datasets,dataset)   
        for j in range(0,1):
            x1 = []
            x2 = []
            for k in range(0,4):
                avg_sig1 = bandpower(signal1[j], sf, bands[k], window)
                avg_sig2 = bandpower(signal2[j], sf, bands[k], window)
                x1.append(avg_sig1)
                x2.append(avg_sig2)
            X.append([x1,x2])
            y.append(count)

    temp = sum(y)
    y = [a/temp for a in y]

    final_X = []
    band11 = 0 
    band21 = 0
    band31 = 0
    band41 = 0
    band12 = 0 
    band22 = 0
    band32 = 0
    band42 = 0

    for i in range(0,5):
        signal1 = X[i][0]
        signal2 = X[i][1]

        band11 += signal1[0]*y[i]
        band21 += signal1[1]*y[i]
        band31 += signal1[2]*y[i]
        band41 += signal1[3]*y[i]
        
        band12 += signal2[0]*y[i]
        band22 += signal2[1]*y[i]
        band32 += signal2[2]*y[i]
        band42 += signal2[3]*y[i]
        
    final_X = [[band11,band21,band31,band41],[band12,band22,band32,band42]]

    return final_X

def GET_SLEEP_STAGE(file1,file2):
    # Store the edf paths in recordings
    recording=[]
    recording.append(file1)
    recording.append(file2)

    # Load the edf as mne.io.raw object
    raws = load_sleep_physionet_raw(recording[0], recording[1])

    # Band-Pass Filtering with Cutout 30Hz
    l_freq, h_freq = None, 30
    raws.load_data().filter(l_freq, h_freq)  # filtering happens in-place

    # Apply windowing and move to pytorch dataset
    all_datasets = [EpochsDataset(*extract_epochs(raws), subj_nb=raws.info['subject_info']['id'], 
                    rec_nb=raws.info['subject_info']['rec_id'], transform=scale) ]

    # Concatenate into a single dataset
    dataset = ConcatDataset(all_datasets)

    # Process EDF for Prediciton
    X = Process(all_datasets,dataset)
    final_X = np.array(X)
    final_X = final_X.reshape(-1,2,4,1)

    # Load the models
    model1 = keras.models.load_model('./CNN1.h5')
    model2 = keras.models.load_model('./CNN2.h5')
    model3 = keras.models.load_model('./CNN3.h5')
    model4 = keras.models.load_model('./LSTM1.h5')
    model5 = keras.models.load_model('./LSTM2.h5')
    model6 = keras.models.load_model('./LSTM3.h5')
    # model1 = pickle.load(open('./CNN1.sav','rb'))
    # model2 = pickle.load(open('./CNN2.sav','rb'))
    # model3 = pickle.load(open('./CNN3.sav','rb'))
    # model4 = pickle.load(open('./LSTM1.sav','rb'))
    # model5 = pickle.load(open('./LSTM2.sav','rb'))
    # model6 = pickle.load(open('./LSTM3.sav','rb'))

    # Predict the Sleep Stage
    y1 = model1.predict(final_X)
    y2 = model2.predict(final_X)
    y3 = model3.predict(final_X)
    y4 = model4.predict(final_X)
    y5 = model5.predict(final_X)
    y6 = model6.predict(final_X)

    # Process the Result
    pred1 = np.argmax(np.round(y1),axis=1)
    pred2 = np.argmax(np.round(y2),axis=1)
    pred3 = np.argmax(np.round(y3),axis=1)
    pred4 = np.argmax(np.round(y4),axis=1)
    pred5 = np.argmax(np.round(y5),axis=1)
    pred6 = np.argmax(np.round(y6),axis=1)

    lst = [pred1,pred2,pred3,pred4,pred5,pred6]
    result = max(lst,key=lst.count)
    score = (lst.count(result)/6)*100

    # Result and Accuracy Score
    return result,score
