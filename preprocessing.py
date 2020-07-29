#@title Data retrival
import os, requests
import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np



def equalise_classes(y, shuffle=False, min_freq=None):
    """
    Equalises the count (the frequency of occurrence) of classes within an array.
    :param y: labels (1d array)
    :param shuffle: if True shuffle the labels
    :param min_freq: pre-defined count
    :return:    y_new: resampled labels array
                idc_new: indices of the new samples given the previous shape
                min_val: minimal count of classes after equalising
    """
    from itertools import groupby
    import pandas as pd
    results = {value: len(list(freq)) for value, freq in groupby(sorted(y))}
    if min_freq == None:
        min_val = min(results.values())
    else:
        min_val = int(min_freq)
    df_idcs = pd.DataFrame(y, columns=['class'])
    df_idcs['Indices'] = list(range(0, len(y)))
    idc_lis = []
    for _ in range(0, len(results.keys())):
        idc_lis.append(df_idcs[df_idcs['class'] == float(_ + 1)].sample(min_val))
    df_idcs_res = pd.concat(idc_lis, axis=0)
    if shuffle:
        idc_res = df_idcs_res['Indices'].values
    else:
        idc_res = df_idcs_res['Indices'].values
        idc_res.sort()
    idc_new = idc_res
    y_new = y[idc_res]
    return y_new, idc_new, min_val


def re_code_bahviour(dat):
    """
    Recodes metadata available in the instance of the dataset into lables.
    :param dat: session instance of the Steinmetz et al. dataset
    :return:    labels (1d array): array containing labels of the trials types with regard to the presented stimuli
                e.i. 1 = higher contrast on the right, 2 = higher contrast on the left, 3 = equal contrast
                correct_v (1d array): array containing labels of the trials types with regard to task performance
                e.i. 0 = incorrect trial, 1 = correct trial
                prop_cor(float): proportion of correct responses in a given session
    """
    contrast_L = dat['contrast_left']
    contrast_R = dat['contrast_right']
    response = dat['response']
    contrast_diff = contrast_L - contrast_R  # negative == higher contrast on the right side
    labels = []
    for _ in range(len(contrast_diff)):
        if contrast_diff[_] < 0:
            labels.append(1)
        elif contrast_diff[_] > 0:
            labels.append(2)
        else:
            labels.append(3)
    labels = np.array(labels)
    correct_v = []
    for _ in range(len(response)):
        if (contrast_diff[_] > 0) & (response[_] > 0):
            correct_v.append(1)
        elif (contrast_diff[_] < 0) & (response[_] < 0):
            correct_v.append(1)
        elif (contrast_diff[_] == response[_]):
            correct_v.append(1)
        else:
            correct_v.append(0)
    prop_cor = np.sum(correct_v) / len(correct_v)
    return np.array(labels), np.array(correct_v), prop_cor


def re_bin_data(data, bin=10):
    return np.array(np.split(data, int(bin), axis=2)).sum(0)


def mean_center_condi(data, labels):
    # mean-centre the data over conditions
    values = np.unique(labels)
    data_centered = []
    for value in values:
        indi = labels == value
        data_re = data[indi, :, :]
        mean = data_re.mean(0)[None, :, :]
        data_centered.append(data_re - mean)
    data_ = np.vstack(data_centered)
    labels_ = np.sort(labels)
    return data_, labels_


def exclude_neurons(data, firing_threshold=0.5):
    data_re = np.array(np.split(data, int(25), axis=2)).sum(0)
    firing_avg = data_re.mean(0)
    idx = firing_avg.mean(1) > firing_threshold
    prop = np.sum(idx) / len(idx)
    plt.figure()
    plt.plot(firing_avg.T)
    plt.ylabel("Firing rate (Hz)")
    plt.xlabel("Time (1s bins)")
    return idx, prop


def get_matched_pops(a1, a2, n_bins=20, geometric_bins=True):
    '''
    Given two spike-count datasets in 'neurons by trials by time bins' format, return a similar
    pair with equal neuron counts and roughly matching distributions of average firing rates.
    Increasing b improves the distributional match but reduces the number of neurons retained.
    args:
    a1, a2 (integer numpy arrays): spike count datasets a la Steinmetz (neurons by trials by time bins)
    n_bins (integer): number of histogram bins to use for the firing-rate matching
    geometric_bins (boolean): if True, space histogram bins evenly on log scale; if False, linear
    returns:
    Tuple (a1_reduced, a2_reduced, a1_cells_idx, a2_cells_idx)
    a1_reduced and a2_reduced are spike count datasets in the same numpy array format as the
    inputs. They have exactly matching shapes and roughly matching firing rate distributions.
    a1_cells_idx and a2_cells_idx are lists of integers. The integers are indices of the
    retained cells in the original datasets.
    '''
    N_trials, N_time_bins = a1.shape[1:]
    assert a1.shape[1:] == a2.shape[1:], 'incompatible datasets'
    # Want to exclude cells whose summed counts (over time bins and trials) falls below a threshold.
    # Here we compute the threshold. Have kept these params out of function sig to avoid clutter.
    t_bin = 0.01  # seconds
    min_freq = 0.5  # Hz
    min_count = min_freq * N_trials * N_time_bins * t_bin
    a1_totals = a1.sum(axis=(1, 2))  # vector of spike counts by a1 cell, summed across bins and trials
    a2_totals = a2.sum(axis=(1, 2))  # ditto for a2
    lumped = np.concatenate((a1_totals, a2_totals))
    bin_func = np.geomspace if geometric_bins else np.linspace
    bins = bin_func(min_count, max(lumped) + 1, n_bins)  # define histogram bins for the distribution matching
    # construct aligned firing rate histograms for a1 and a2
    his1 = np.histogram(a1_totals, bins=bins)[0]
    his2 = np.histogram(a2_totals, bins=bins)[0]
    n_a1, n_a2 = a1.shape[0], a2.shape[0]  # numbers of cells in original a1 and a2 populations
    matched_a1_cells = []
    matched_a2_cells = []
    for i in range(len(bins) - 1):
        mn, mx = bins[i], bins[i + 1]
        a1_candidates = [j for j in range(n_a1) if (a1_totals[j] >= mn and a1_totals[j] < mx)]
        a2_candidates = [j for j in range(n_a2) if (a2_totals[j] >= mn and a2_totals[j] < mx)]
        quota = min(len(a1_candidates), len(a2_candidates))
        matched_a1_cells.extend(np.random.choice(a1_candidates, size=quota, replace=False))
        matched_a2_cells.extend(np.random.choice(a2_candidates, size=quota, replace=False))
    return a1[matched_a1_cells, :, :], a2[matched_a2_cells, :, :], matched_a1_cells, matched_a2_cells


def drop_quiet_cells(spks, min_av_spks, verbose=True):
    '''
    Given a spike-count dataset in 'neurons by trials by time bins' format, return a trimmed
    dataset with low-activity cells removed.
    args:
    spks (integer numpy array): spike count dataset a la Steinmetz (neurons by trials by time bins)
    min_av_spks (float): threshold spikes-per-bin rate (averaged across trials and times) below
    which neurons are dropped
    verbose (boolean): if True, function prints number of dropped cells to screen
    returns:
    integer numpy array: the result of removing low-activity cells from spks
    '''
    av_spks_by_cells = spks.mean(axis=(1, 2))
    cell_mask = av_spks_by_cells >= min_av_spks
    if verbose:
        print('Dropped', sum(np.logical_not(cell_mask)), 'of the original', len(cell_mask), 'cells')
    return spks[cell_mask, :, :]


def calculate_psths(spikes, vis_left=dat['contrast_left'], vis_right=dat['contrast_right']):
    """
    get spikes and substracts the corresponding psth that accounts for this particular trial
    type, meaning particular contrast valuse
    """

    cases = np.array([(i, j) for i in np.unique(vis_right) for j in np.unique(vis_left)])
    psths = np.zeros((len(cases), a.shape[0], a.shape[2]))
    case_cnt = np.zeros((len(cases),))
    for i in range(a.shape[1]):
        for caseid, case in enumerate(cases):
            if (np.array([vis_right[i], vis_left[i]]) == case).all():
                psths[caseid, :] += a[:, i, :]
                case_cnt[caseid] += 1
    psths = np.array([psths[i, :, :]/j for i, j in enumerate(case_cnt)])
    for i in range(a.shape[1]):
        for caseid, case in enumerate(cases):
            if (np.array([vis_right[i], vis_left[i]]) == case).all():
                a[:,i,:] -= psths[caseid, :, :]
    return a

def spike_from_stimulus_type(spikes, contrast, vis_left=dat['contrast_left'], vis_right=dat['contrast_right']):
    '''
    get spikes and a particular contrast type and returns all the trials with this particular contrast values
    '''
    cases = np.array([(i, j) for i in np.unique(vis_right) for j in np.unique(vis_left)])
    trials = []
    for i in range(a.shape[1]):
        if (np.array([vis_right[i], vis_left[i]]) == contrast).all():
            trials += [i]
    trials = np.array(trials)
    return spikes[:, trials, :]
