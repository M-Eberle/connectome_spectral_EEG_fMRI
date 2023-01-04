# %% [markdown]

### last steps
# - load data
# - sort data (participants, regions)
# - transform data onto graph
# - check that participant's indices in SC, fMRI, and EEG align
# - symmetry of SCs? --> use +transpose for now
# - interpolate fMRI
# - look at data
#   - activity, harmonics, power
#   - correlations between fMRI & EEG
# - sanity checks for EEG data
#   - EEG power, mean over participants
#   - EEG frequency band
#   - EEG-fMRI -> alpha regressor, alpha power band vs fMRI timeseries
# - compare graph vs vertex domain: corr EEG-fMRI
# - plot signal on graph
# - compare patterns between participants
# - data in numpy arrays instead of lists
# - general refactoring
# - scale timeseries before applying similarity measures
# - compare timeseries EEG & fMRI (e.g. lower vs higehr half of harmonics)
# - Fischer Transform instead of mean for correlations
# - use EEG outlier detection
# - get random TVGs
# next steps
# - save similarity measures as Data class attributes (save time generating JET)
# - decide on suitable gamma for JET
# - compare similarity measure results to random signals/graphs
#### other ToDos
# - fix time axis ticks in all plots / both axes ticks in heatmaps
# - compare scipy.interpolate.interp1d & scipy.signal.resample


# %%
from helpers import *

# %%
# currently, some plots/calculations only work for individual SC matrices
SC_mode = "ind"  # 'mean' or 'ind'

# looping over all participants for everything takes too long
data_ind = Data(SC_mode=SC_mode, loop_participants=False)

# %%
# overview plots
data_ind.plot_signal()
data_ind.plot_signal_single_domain()
data_ind.plot_domain()
data_ind.plot_power_stem_cum()
data_ind.plot_power_corr()

# sanity checks for EEG
# sanity check 1
data_ind.plot_power_mean_corr()
data_ind.plot_power_mean_stem_cum()
# sanity check 2
data_ind.plot_EEG_freq_band()
# sanity check 3
data_ind.get_alpha_corrs()
data_ind.plot_alpha_corrs()
data_ind.plot_alpha_corrs_on_graph()

# hypotheses
# vertex vs graph
data_ind.get_vertex_vs_graph()
data_ind.plot_vertex_vs_graph()
# harmonic's power
data_ind.get_lower_vs_upper_harmonics()
# similarity bewteen participants
data_ind.get_TVG()
data_ind.get_GE()
data_ind.get_JET()
# compare random TVG matrices to participant's ?
data_ind.get_random_TVG(random_weights=True)
data_ind.get_random_TVG(random_weights=False)
# TVG between eigenvectors
data_ind.TVG_evecs()


# %%
# ________________________________________________
# additional artifact subspace reconstruction????
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
import numpy as np
import matplotlib.pyplot as plt

# %%
all_powers = np.empty((data_ind.N_regions, data_ind.N))
for participant in np.arange(data_ind.N):
    sfreq = 200
    raw = data_ind.EEG_timeseries[:, :, participant]
    # Train on a clean portion of data
    asr = ASR(method="euclid")
    train_idx = np.arange(0 * sfreq, 30 * sfreq, dtype=int)
    _, sample_mask = asr.fit(raw[:, train_idx])

    # Apply filter using sliding (non-overlapping) windows
    X = sliding_window(raw, window=int(sfreq), step=int(sfreq))
    Y = np.zeros_like(X)
    for i in range(X.shape[1]):
        Y[:, i, :] = asr.transform(X[:, i, :])

    raw = X.reshape(data_ind.N_regions, -1)  # reshape to (n_chans, n_times)
    clean = Y.reshape(data_ind.N_regions, -1)

    # compare 'raw' vs cleaned signal
    plt.plot(raw[1, :])
    plt.plot(clean[1, :])
    plt.xlabel("timepoints")
    plt.ylabel("signal")
    plt.title(f"cleaned vs 'raw' region 1 for {participant + 1}")
    plt.show()

    # compare 'raw' vs cleaned power
    EEG_power = (clean) ** 2
    EEG_power = EEG_power / np.sum(EEG_power, 0)[np.newaxis, :]
    power_mean = np.mean(EEG_power, 1)
    plt.stem(power_mean)
    plt.xlabel("harmonic")
    plt.ylabel("signal strength")
    plt.title(
        f"cleaned normalized graph frequency domain for participant {participant + 1}\n (mean over time, normalized per timepoint)"
    )
    plt.show()

    all_powers[:, participant] = power_mean

plt.stem(np.mean(all_powers, 1))
plt.xlabel("harmonic")
plt.ylabel("signal strength")
plt.title(
    f"cleaned normalized graph frequency domain\n (mean over time and participants, normalized per timepoint)"
)
plt.show()
