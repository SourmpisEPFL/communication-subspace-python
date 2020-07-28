#@title Data retrival
import os, requests
import matplotlib.pyplot as plt
from scipy.io import savemat
import matlab.engine
import numpy as np
from preprocessing import *

fname = []
for j in range(3):
  fname.append('steinmetz_part%d.npz'%j)
url = ["https://osf.io/agvxh/download"]
url.append("https://osf.io/uv3mw/download")
url.append("https://osf.io/ehmw2/download")
for j in range(len(url)):
  if not os.path.isfile(fname[j]):
    try:
      r = requests.get(url[j])
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname[j], "wb") as fid:
          fid.write(r.content)

#@title Data loading
alldat = np.array([])
for j in range(len(fname)):
  alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))
# select just one of the recordings here. 11 is nice because it has some neurons in vis ctx.
dat = alldat[11]

print(dat.keys())



#@title Example
spks = np.moveaxis(dat['spks'], [0, 1, 2], [1, 0, 2])  # trials x neurons x time bins.
brain_areas = dat['brain_area']
labels, correct_v, prop_cor = re_code_bahviour(dat)
data = spks[:, brain_areas == 'VISam', :]
data_rebin = re_bin_data(data, bin=10)
labels_new, idc, _ = equalise_classes(labels)
data_new = data_rebin[idc, :, :]
data_, labels_ = mean_center_condi(data_new, labels_new)
idx, prop = exclude_neurons(data, 0.5)
data_preproc = data_[:, idx, :]
plt.figure()
plt.imshow(data_preproc.mean(1))
plt.xlabel("Time (bins)")
plt.ylabel("Trials")
plt.title("Residual firing rate averaged over neurons")


# demo using session 11 data (assuming 'dat' has value assigned in first code blocks in 'Data curation')
dat = alldat[14]
N_cells = dat['spks'].shape[0]
areas = set(dat['brain_area'])
cells_by_area = {area: [i for i in range(N_cells) if dat['brain_area'][i]==area] for area in areas}
a1 = drop_quiet_cells(dat['spks'][cells_by_area['VPM'],:,:],0.005)
a2 = drop_quiet_cells(dat['spks'][cells_by_area['GPe'],:,:],0.005)


# splits a1 into a1 target and a1 source
a1_red, a2_red, a1_red_idx, a2_red_idx = get_matched_pops(a1, a2, n_bins=30)
def run_matlab_script(session,area_1,area_2):
    dat = alldat[session]
    N_cells = dat['spks'].shape[0]
    areas = set(dat['brain_area'])
    cells_by_area = {area: [i for i in range(N_cells) if dat['brain_area'][i] == area] for area in areas}
    # Drop cells that
    a1 = drop_quiet_cells(dat['spks'][cells_by_area[area_1], :, :], 0.005)
    a2 = drop_quiet_cells(dat['spks'][cells_by_area[area_2], :, :], 0.005)
    # splits a1 into a1 target and a1 source
    a1_red, a2_red, a1_red_idx, a2_red_idx = get_matched_pops(a1, a2, n_bins=30)
    a1_unmatched = np.delete(a1,a1_red_idx,axis = 0) # This is a1 source
    # Reshape to put into matlab
    a1_unmatched_reshape = a1_unmatched.reshape(a1_unmatched.shape[0],a1_unmatched.shape[1]*a1_unmatched.shape[2]).T
    a1_red_reshape = a1_red.reshape(a1_red.shape[0],a1_red.shape[1]*a1_red.shape[2]).T
    a2_red_reshape = a2_red.reshape(a2_red.shape[0],a2_red.shape[1]*a2_red.shape[2]).T

    # Open matlab engine, run script.
    eng = matlab.engine.start_matlab()
    highestdim = np.min([a2_red_reshape.shape[1],
    a1_red_reshape.shape[1],
    a1_unmatched_reshape.shape[1]])
    dimensions = matlab.double(np.arange(1,highestdim).tolist())
    X = matlab.double(a1_unmatched_reshape.tolist())
    Y_V1 = matlab.double(a1_red_reshape.tolist())
    Y_V2 = matlab.double(a2_red_reshape.tolist())

    cvlossself, optdimself, cvlosscross, optdimcross = eng.figure2(
    X, Y_V1, Y_V2, dimensions, nargout=4)
    cvlossself, cvlosscross = np.array(cvlossself), np.array(cvlosscross)

    eng.quit()

    return cvlossself, optdimself, cvlosscross, optdimcross

run_matlab_script(11,'SUB','DG')
asd = 1
