import numpy as np

def rebin(spks, n):
    '''
    Coarsens the time binning of a Steinmetz-format spike-count dataset
    by a factor of n. For example, if the original dataset has 10 ms bins
    and n=10, the returned array will have 100 ms bins.

    If n does not exactly divide spks.shape[2], then the 'remainder cells'
    of spks, at the rightmost end of the 2-axis, are dropped.

    args:
    spks (numpy array): spike-count dataset (cells by trials by time bins)
    shape
    n (integer): number of old time bins per new time bin

    returns:
    a spike-count datasets in same format as input, but with only
    spks.shape[2] // n time bins.
    '''
    cells, trials, old_bins = spks.shape
    new_bins = old_bins // n
    spks_rebinned = np.empty((cells, trials, new_bins))
    for i in range(new_bins):
        spks_rebinned[:,:,i] = spks[:,:,i:(i+n)].sum(axis=2)
    return spks_rebinned

def drop_quiet_cells(spks, min_av_spks, verbose=True):
    '''
    Given a spike-count dataset in 'cells by trials by time bins' format, return a trimmed
    dataset with low-activity cells removed.

    args:
    spks (integer numpy array): spike count dataset a la Steinmetz (cells by trials by time bins)
    min_av_spks (float): threshold spikes-per-bin rate (averaged across trials and times) below
    which cells are dropped
    verbose (boolean): if True, function prints number of dropped cells to screen

    returns:
    integer numpy array: the result of removing low-activity cells from spks
    '''

    av_spks_by_cells = spks.mean(axis=(1,2))
    cell_mask = av_spks_by_cells >= min_av_spks
    if verbose:
        print('Dropped', sum(np.logical_not(cell_mask)), 'of the original', len(cell_mask), 'cells')
    return spks[cell_mask, :, :]

def get_matched_pops(a1, a2, n_hist_bins, geometric_bins=True, verbose=True):
    '''
    Given two spike-count datasets in 'cells by trials by time bins' format,
    return a corresponding pair with equal neuron counts and roughly matching
    distributions of average firing rates.

    Increasing n_hist_bins improves the distributional match but reduces the
    number of neurons retained.

    args:

    a1, a2 (integer numpy arrays): spike count datasets a la Steinmetz (neurons
    by trials by time bins)
    n_hist_bins (integer): number of histogram bins to use for the firing-rate
    matching
    geometric_bins (boolean): if True, space histogram bins evenly on log scale;
    if False, a linear scale is used
    verbose (boolean): if True, function prints info on population sizes

    returns:

    Tuple (a1_reduced, a2_reduced, a1_cells_idx, a2_cells_idx)
    a1_reduced and a2_reduced are spike count datasets in the same numpy array
    format as the inputs. They have exactly matching shapes and roughly matching
    firing rate distributions.
    a1_cells_idx and a2_cells_idx are lists of integers. The integers are
    indices of the retained cells in the original datasets.
    '''

    N_trials, N_time_bins = a1.shape[1:]
    assert a1.shape[1:] == a2.shape[1:], 'incompatible datasets'

    a1_totals = a1.sum(axis=(1,2)) # vector of spike counts by a1 cell, summed across bins and trials
    a2_totals = a2.sum(axis=(1,2)) # ditto for a2
    lumped = np.concatenate((a1_totals,a2_totals))

    min_count = min(lumped) # this will be left-hand edge of first histogram bin
    assert not (geometric_bins and min_count == 0), "can't use geometric spacing when silent cells present"

    bin_func = np.geomspace if geometric_bins else np.linspace
    bins = bin_func(min_count, max(lumped)+1, n_hist_bins) # define histogram bins for the distribution matching

    # construct aligned firing rate histograms for a1 and a2
    his1 = np.histogram(a1_totals, bins=bins)[0]
    his2 = np.histogram(a2_totals, bins=bins)[0]

    n_a1, n_a2 = a1.shape[0], a2.shape[0] # numbers of cells in original a1 and a2 populations

    matched_a1_cells = []
    matched_a2_cells = []

    for i in range(len(bins)-1):
        mn, mx = bins[i], bins[i+1]
        a1_candidates = [j for j in range(n_a1) if (a1_totals[j] >= mn and a1_totals[j] < mx)]
        a2_candidates = [j for j in range(n_a2) if (a2_totals[j] >= mn and a2_totals[j] < mx)]
        quota = min(len(a1_candidates), len(a2_candidates))
        matched_a1_cells.extend(np.random.choice(a1_candidates, size=quota, replace=False))
        matched_a2_cells.extend(np.random.choice(a2_candidates, size=quota, replace=False))

    if verbose:
        print('Pre-match populations:', a1.shape[0], a2.shape[0])
        print('Matched populations:', len(matched_a1_cells), len(matched_a2_cells))
        print('Source population:', a1.shape[0] - len(matched_a1_cells))

    return a1[matched_a1_cells,:,:], a2[matched_a2_cells,:,:], matched_a1_cells, matched_a2_cells
