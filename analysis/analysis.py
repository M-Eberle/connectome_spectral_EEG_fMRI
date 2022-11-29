# %%
from helpers import *

# %% [markdown]

### last steps
# - load data
# - transform data onto graph
# - check that participant's indices in SC, fMRI, and EEG align
# - symmetry of SCs? --> use +transpose for now
# - interpolate fMRI
# - look at data
#   - activity, harmonics, power
#   - correlations between fMRI & EEG
### next steps
# - sanity checks for EEG data
# - compare graph vs vertex domain: corr EEG-fMRI
# - compare timeseries EEG & fMRI (e.g. lower vs higehr half of harmonics)
# - compare patterns between participants
#### other ToDos
# - fix time axis in all plots
# - save plots
# - plot signal on graph nodes
# - data in numpy arrays instead of lists
# - compare scipy.interpolate.interp1d & scipy.signal.resample

# %%
SC_path = "../data/empirical_structural_connectomes/SCs.mat"
fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
EEG_data_path = "../data/empirical_source_activity/source_activity.mat"

regionsMap = sio.loadmat("../data/empirical_source_activity/regionsMap.mat")[
    "regionsMap"
][:, 0]
HRF_resEEG = sio.loadmat("../data/HRF_200Hz.mat")[
    "HRF_resEEG"
]  # hemodynamic response function

# %%
ex_participant = 1
ex_harmonic = 5
ex_region = 5
sampling_freq = 200  # Hz, sampling frequency for EEG
mode = "mean"  # or mean

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
# restructure data as numpy array
EEG_power = np.empty((68, N))
fMRI_power = np.empty((68, N))
for participant in np.arange(N):
    EEG_power[:, participant] = power_mean(trans_EEG_timeseries, participant, "EEG")
    fMRI_power[:, participant] = power_mean(trans_fMRI_timeseries, participant, "fMRI")

# %%
# cumulative power for example participant
plot_cum_power(EEG_power[:, ex_participant], ex_participant, "EEG")
plot_cum_power(fMRI_power[:, ex_participant], ex_participant, "fMRI")
# %%
# EEG sanity check 1: EEG power, mean over participants
# ________________________
# find mean power over all participants ( & over time)
power_mean_EEG = np.mean(EEG_power, 1)
power_mean_fMRI = np.mean(fMRI_power, 1)
# cumulative power, mean over participants (plot titles not correct)
plot_cum_power(power_mean_EEG, ex_participant, "EEG")
plot_cum_power(power_mean_fMRI, ex_participant, "fMRI")

# %%
# EEG sanity check 2: EEG frequency band
# ________________________
# NO means (separate for each region and participant)

plot_fft_welch(EEG_timeseries[ex_participant][ex_region, :], sampling_freq)

# alpha-activity very prominent

# %%
# EEG sanity check 3: EEG-fMRI
# ________________________

# filter to alpha band: 8.5-12 Hz
# maybe repeat for delta (0.1–4 Hz), theta (4.5–8 Hz), beta (12.5–36 Hz), gamma (36.5–100 Hz) bands
# filter each region for each participant separately
filtered_EEG = butter_bandpass_filter(EEG_timeseries[ex_participant], 8.5, 12, 200)

freqs, psd_before = plot_fft_welch(
    EEG_timeseries[ex_participant][ex_region, :], sampling_freq
)
freqs, psd_filtered = plot_fft_welch(filtered_EEG[ex_region, :], sampling_freq)
# cut out frequency and reverse transform????


# BOLD signal can be sufficiently described as the convolution between
# a linear combination of the power profile within individual frequency bands
# with a hemodynamic response function (HRF)

# negative correlation between the BOLD signal and the average power time series
# within the alpha band (8--12 Hz) --> only on some electrodes though and we have regions instead?

# find spectral profiles corresponding to different frequency band
# %%

# find data points that correspond to fMRI data points
# OR use interpolated fMRI?

# Hilbert-Transform for band limited power for all time points
# OR get spectra for timepoints from before for each volume, then mean over alpha band? (de Munck ?)
# Hilbert transform (Matlab code, gibt es sicher auch für python): Xanalytic = hilbert(data)
# pow_env = abs(Xanalytic(11:end-10)) % Anfang und Ende abschneiden wegen transients
# more info: https://www.gaussianwaves.com/2017/04/extract-envelope-instantaneous-phase-frequency-hilbert-transform/
# EEG_env = np.abs(sg.hilbert(filtered_EEG, axis=1))

# mean for each volume --> power timeseries in same sampling freq as fMRI

# for each region, correlate EEG power timeseries with fMRI

# repeat for different time shifts

# %%
alpha_reg, alpha_reg_filt = compute_alpha_regressor(
    EEG_timeseries, ex_participant, regionsMap, HRF_resEEG, sampling_freq
)


# %%
alpha_timeseries, alpha_timeseries_filt = compute_power_timeseries(
    EEG_timeseries, ex_participant, regionsMap, sampling_freq
)
