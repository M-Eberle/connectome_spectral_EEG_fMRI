# %%
from helpers import *


# %% [markdown]

### last steps
# - load data
# - transform data onto graph
# - check that participant's indices in SC, fMRI, and EEG align
# - symmetry of SCs? --> use +transpose for now
### next steps
# - compare fMRI & EEG signal with individual SCs
#   - plot power over smoothness per participant
#   - nr. of harmonics needed to recreate fMRI/EEG signal --> cumulative power from Glomb et al. (2020) Fig. 2
#   - compare patterns between participants (correlation matrices?)
# - compare fMRI & EEG signal with average SC
#   - averaged correlation matrix ?
#   - ? plot power over smoothness per participant
#### other ToDos
# - fix time axis in all plots
# - save plots
# - plot signal on graph nodes
# - data in numpy arrays instead of lists

# %%
SC_path = "../data/empirical_structural_connectomes/SCs.mat"
fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
EEG_data_path = "../data/empirical_source_activity/source_activity.mat"

ex_participant = 1
ex_harmonic = 5
ex_region = 5
mode = "mean"  # or mean

# %%
if mode == "ind":
    (
        SC_weights,
        EEG_timeseries,
        trans_EEG_timeseries,
        fMRI_timeseries,
        fMRI_timeseries_interp,
        trans_fMRI_timeseries,
        N,
        N_regions,
        EEG_timesteps,
    ) = get_data_ind_SCs(SC_path, EEG_data_path, fMRI_data_path)
else:
    (
        SC_weights,
        EEG_timeseries,
        trans_EEG_timeseries,
        fMRI_timeseries,
        fMRI_timeseries_interp,
        trans_fMRI_timeseries,
        N,
        N_regions,
        EEG_timesteps,
    ) = get_data_mean_SC(SC_path, EEG_data_path, fMRI_data_path)

N = len(trans_EEG_timeseries)
plot_ex_interp(fMRI_timeseries, fMRI_timeseries_interp, ex_participant, ex_harmonic)


# %%
plot_ex_signal_EEG_fMRI(
    EEG_timeseries, fMRI_timeseries_interp, ex_participant, "region"
)
plot_ex_signal_EEG_fMRI(
    trans_EEG_timeseries, trans_fMRI_timeseries, ex_participant, "harmonic"
)
# %%
plot_ex_regions_harmonics(EEG_timeseries, trans_EEG_timeseries, ex_participant, "EEG")
plot_ex_regions_harmonics(
    fMRI_timeseries_interp, trans_fMRI_timeseries, ex_participant, "fMRI"
)

# %%
EEG_power_norm = power_norm(trans_EEG_timeseries, ex_participant)
fMRI_power_norm = power_norm(trans_fMRI_timeseries, ex_participant)

plot_ex_power_EEG_fMRI(EEG_power_norm, fMRI_power_norm, ex_participant)
plot_power_corr(EEG_power_norm, fMRI_power_norm, ex_participant)

# %%
plot_ex_signal_fMRI_EEG_one(
    EEG_timeseries, fMRI_timeseries_interp, ex_participant, ex_region, "region"
)
plot_ex_signal_fMRI_EEG_one(
    trans_EEG_timeseries, trans_fMRI_timeseries, ex_participant, ex_harmonic, "harmonic"
)

# %%
# split into highest vs lowest 15 harmonics & do similarity analysis of correlation matrices?
regions_corr = ex_EEG_fMRI_corr(
    EEG_timeseries, fMRI_timeseries_interp, ex_participant, "region"
)
harmonics_corr = ex_EEG_fMRI_corr(
    trans_EEG_timeseries, trans_fMRI_timeseries, ex_participant, "harmonic"
)

# %%
# find mean power over all participants ( & over time)
EEG_power = np.empty((68, N))
fMRI_power = np.empty((68, N))
for participant in np.arange(N):
    EEG_power[:, participant] = power_mean(trans_EEG_timeseries, participant, "EEG")
    fMRI_power[:, participant] = power_mean(trans_fMRI_timeseries, participant, "fMRI")

# %%
# EEG sanity check 1: EEG cumulative power (mean over time)
# ________________________
power_mean_EEG = np.mean(EEG_power, 1)
power_mean_fMRI = np.mean(fMRI_power, 1)
# plot titles not correct
plt.stem(power_mean_EEG)
plot_cum_power(power_mean_EEG, ex_participant, "EEG")
plt.stem(power_mean_fMRI)
plot_cum_power(power_mean_fMRI, ex_participant, "fMRI")
# ___________________________

# %%
plt.stem(EEG_power[:, ex_participant])
plot_cum_power(EEG_power[:, ex_participant], ex_participant, "EEG")

plt.stem(fMRI_power[:, ex_participant])
plot_cum_power(fMRI_power[:, ex_participant], ex_participant, "fMRI")

# %%
# EEG sanity check 2: EEG frequency band
# ________________________
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# %%
# mean over participants
EEG_t = np.empty((68, 259184, N))
for participant in np.arange(N):
    EEG_t[:, :, participant] = EEG_timeseries[participant]

# %%
# keep data from all brain regions, concatenate to 1 vector?
EEG_mean = np.mean(EEG_t, 2).flatten()
N = len(EEG_mean)

# method 1
dt = 0.0005  # Define the sampling interval
T = N * dt  # Define the total duration of the data
xf = np.fft.fft(EEG_mean - EEG_mean.mean())  # Compute Fourier transform of x
Sxx = 2 * dt**2 / T * (xf * xf.conj())  # Compute spectrum
Sxx = Sxx[: int(len(EEG_mean) / 2)]  # Ignore negative frequencies

df = 1 / T  # Determine frequency resolution
fNQ = 1 / dt / 2  # Determine Nyquist frequency
faxis = np.arange(0, fNQ, df)  # Construct frequency axis

plt.plot(faxis, Sxx.real)  # Plot spectrum vs frequency
plt.xlim([0, 100])  # Select frequency range
plt.xlabel("Frequency [Hz]")  # Label the axes
plt.ylabel("Power [$\mu V^2$/Hz]")
plt.show()


# %%
# method 2
# Define sampling frequency and time vector
sf = 200  # Hz
time = np.arange(EEG_mean.size) / sf
from scipy import signal

# Define window length (4 seconds)
win = 2 * sf
freqs, psd = signal.welch(EEG_mean, sf, nperseg=win)
plt.plot(freqs, 1 / freqs, label="1/f")
# plt.plot(freqs, psd)
print(1 / freqs)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectral density (V^2 / Hz)")
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, freqs.max()])
plt.legend()
# sns.despine()
# ___________________________
