import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as sg
import matplotlib.cm as mcm

from helpers.overview_plots import corrcoef2D

#%% [markdown]
# EEG sanity check 3: EEG-fMRI
# ________________________

# filter to alpha band: 8.5-12 Hz
# maybe repeat for delta (0.1–4 Hz), theta (4.5–8 Hz), beta (12.5–36 Hz), gamma (36.5–100 Hz) bands
# filter each region for each participant separately


# BOLD signal can be sufficiently described as the convolution between
# a linear combination of the power profile within individual frequency bands
# with a hemodynamic response function (HRF)

# negative correlation between the BOLD signal and the average power time series
# within the alpha band (8--12 Hz) --> only on some electrodes though and we have regions instead?

# find spectral profiles corresponding to different frequency band

# find data points that correspond to fMRI data points

# Hilbert-Transform for band limited power for all time points
# OR get spectra for timepoints from before for each volume, then mean over alpha band? (de Munck ?)
# Hilbert transform (Matlab code, gibt es sicher auch für python): Xanalytic = hilbert(data)
# pow_env = abs(Xanalytic(11:end-10)) % Anfang und Ende abschneiden wegen transients
# more info: https://www.gaussianwaves.com/2017/04/extract-envelope-instantaneous-phase-frequency-hilbert-transform/
# EEG_env = np.abs(sg.hilbert(filtered_EEG, axis=1))

# mean for each volume --> power timeseries in same sampling freq as fMRI
# ??????


# for each region, correlate EEG power timeseries with fMRI

# repeat for different time shifts
# %%


def compute_alpha_regressor(
    EEG_timeseries, ex_participant, regionsMap, HRF_resEEG, sampling_freq
):
    """
    adapted from Schirner et al. (2018) implementation https://github.com/BrainModes/The-Hybrid-Virtual-Brain/blob/master/MATLAB/compute_alpha_regressor.m
    convoluted a-band power fluctuation of injected EEG source activity with canonical hemodynamic response function = alpha regressor

    arguments:
        EEG_timeseries: EEG timeseries
        ex_participant: example participant index
        regionsMap: [68 x 1] vector that contains the region sorting of source
                    activity as outputted by Brainstorm
        HRF_resEEG: [1 x 6498] vector that contains the hemodynamic
                    response function sampled at 200 Hz
        sampling_freq: EEG sampling frequency
    returns:
        alpha_reg: alpha regressor (edges are discarded due to edge
                    effects from filtering and from convolution with HRF)
        alpha_reg_filt: filtered alpha regressor (edges are discarded due to edge
                    effects from filtering and from convolution with HRF)
    """
    source_activity = EEG_timeseries[ex_participant]
    # Sorting of Desikan-Killiany atlas regions in SC matrices
    # use sorting also for everything else?
    SCmat_sorting = np.concatenate(
        np.array(
            (
                np.arange(1001, 1003 + 1),
                np.arange(1005, 1035 + 1),
                np.arange(2001, 2003 + 1),
                np.arange(2005, 2035 + 1),
            ),
            dtype=object,
        )
    ).ravel()

    # Generate butterworth filter for alpha range and for resting-state
    # slow oscillations range
    order = 1  # why not 5?
    TR = 1.94  # BOLD sampling rate
    alpha = np.array((8.5, 12))
    Wn = alpha / (sampling_freq / 2)  # devide by nyquist frequency
    b_hi, a_hi = sg.butter(order, Wn, btype="band")
    # is this also bandpass? what are the values?
    # for EEG activity on longer timescales?
    b_lo, a_lo = sg.butter(
        order, 0.1 / ((1 / TR) / 2)
    )  # also divide by 2 if only 1 value (not bandpass but low/highpass!)

    # Initialize output arrays (shorter than full fMRI time series to
    # discard edge effects from filtering and convolution with HRF)
    alpha_reg = np.zeros((651, 68))
    alpha_reg_filt = np.zeros((651, 68))

    # Iterate over regions
    for region in np.arange(68):
        # Get SC matrix sorting
        regindSAC = np.argwhere(regionsMap == SCmat_sorting[region])
        region_ts = source_activity[regindSAC, :]

        # Filter in alpha range
        # Zero-phase digital filtering, padlen changes from matlab to python
        region_ts_filt = sg.filtfilt(
            b_hi,
            a_hi,
            region_ts,
            padtype="odd",
            padlen=3 * (max(len(b_hi), len(a_hi)) - 1),
        )

        # Hilbert transform to get instantaneous amplitude
        region_ts_filt_hilb = sg.hilbert(region_ts_filt)
        inst_ampl = np.abs(region_ts_filt_hilb)[0, :, :]

        # Convolution with HRF
        # indexing adapted? no second dimension before
        inst_ampl_HRF = sg.convolve(
            inst_ampl[:, 100 : len(inst_ampl) - 100], HRF_resEEG, "valid"
        )[0, :]

        # Downsample to BOLD sampling rate (TR = 1.94 s)
        N_downsample = 388
        HRF_idx = np.arange(100, len(inst_ampl_HRF) - 100)
        downsample_idx = np.empty((651)).astype(int)
        downsample_idx[0] = 0
        for i in np.arange(651 - 1):
            downsample_idx[i + 1] = downsample_idx[i] + N_downsample

        # print(downsample_idx)
        alpha_reg[:, region] = inst_ampl_HRF[HRF_idx][downsample_idx]

        alpha_reg_filt[:, region] = sg.filtfilt(
            b_lo,
            a_lo,
            alpha_reg[:, region],
            padtype="odd",
            padlen=3 * (max(len(b_hi), len(a_hi)) - 1),
        )

    return alpha_reg.T, alpha_reg_filt.T


def compute_power_timeseries(EEG_timeseries, ex_participant, regionsMap, sampling_freq):
    """
    adapted from Schirner et al. (2018) implementation https://github.com/BrainModes/The-Hybrid-Virtual-Brain/blob/master/MATLAB/compute_alpha_regressor.m
    a-band power fluctuation of injected EEG source activity

    arguments:
        EEG_timeseries: EEG timeseries
        ex_participant: example participant index
        regionsMap: [68 x 1] vector that contains the region sorting of source
                    activity as outputted by Brainstorm

    returns:
        alpha_reg: alpha regressor (edges are discarded)
        alpha_reg_filt: filtered alpha regressor (edges are discarded)
    """
    source_activity = EEG_timeseries[ex_participant]
    # Sorting of Desikan-Killiany atlas regions in SC matrices
    SCmat_sorting = np.concatenate(
        np.array(
            (
                np.arange(1001, 1003 + 1),
                np.arange(1005, 1035 + 1),
                np.arange(2001, 2003 + 1),
                np.arange(2005, 2035 + 1),
            ),
            dtype=object,
        )
    ).ravel()

    # Generate butterworth filter for alpha range and for resting-state
    # slow oscillations range
    order = 1  # why not 5?
    TR = 1.94  # BOLD sampling rate
    alpha = np.array((8.5, 12))
    Wn = alpha / (sampling_freq / 2)  # devide by nyquist frequency
    b_hi, a_hi = sg.butter(order, Wn, btype="band")
    # is this also bandpass? what are the values?
    # for EEG activity on longer timescales?
    b_lo, a_lo = sg.butter(
        order, 0.1 / ((1 / TR) / 2)
    )  # also divide by 2 if only 1 value (low/highpass, not bandpass?)

    # Initialize output arrays (shorter than full fMRI time series to
    # discard edge effects from filtering and convolution with HRF)
    alpha_power = np.zeros((651, 68))
    alpha_power_filt = np.zeros((651, 68))

    # Iterate over regions
    for region in np.arange(68):
        # Get SC matrix sorting
        regindSAC = np.argwhere(regionsMap == SCmat_sorting[region])

        # no sorting???? because I correlate with unsorted data??
        # region_ts = source_activity[regindSAC, :]
        region_ts = source_activity[region, :]

        # Filter in alpha range
        # Zero-phase digital filtering, padlen changes from matlab to python
        region_ts_filt = sg.filtfilt(
            b_hi,
            a_hi,
            region_ts,
            padtype="odd",
            padlen=3 * (max(len(b_hi), len(a_hi)) - 1),
        )

        # Hilbert transform to get instantaneous amplitude
        region_ts_filt_hilb = sg.hilbert(region_ts_filt)
        inst_ampl = np.abs(region_ts_filt_hilb)  # [0, 0, :]

        # Downsample to BOLD sampling rate (TR = 1.94 s)
        N_downsample = 388
        sub_idx = np.arange(
            100, len(inst_ampl) - 100
        )  # no sub_idx because no convolution --> keep ends?
        downsample_idx = np.empty((651)).astype(int)
        downsample_idx[0] = 0
        for i in np.arange(651 - 1):
            downsample_idx[i + 1] = downsample_idx[i] + N_downsample

        alpha_power[:, region] = inst_ampl[sub_idx][downsample_idx]

        alpha_power_filt[:, region] = sg.filtfilt(
            b_lo,
            a_lo,
            alpha_power[:, region],
            padtype="odd",
            padlen=3 * (max(len(b_hi), len(a_hi)) - 1),
        )

    return alpha_power.T, alpha_power_filt.T


def normalize_data(data):
    """
    normalizes data between 0 and 1
    arguments:
        data: data to be normalized
    return:
        normalized data
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_alpha_reg_power_fMRI(
    alpha,
    alpha_filt,
    fMRI_timeseries,
    corr,
    mode,
    ex_participant,
    shift,
    uncut,
    max_shift,
):
    """
    Plots alpha component and filtered alpha component vs fMRI timeseries for the region with the
    highest absolute correlation for one participant.
    arguments:
        alpha: alpha regressor or power timeseries
        alpha_filt: filtered alpha regressor or power timeseries
        fMRI_timeseries: fMRI timeseries
        corr_final: array of correlation of alpha component and fMRI
        mode: mode of alpha component ('reg' or 'power') for plotting
        ex_participant: example participant index
        shift: shift in fMRI for exPparticipant (to fit length of alpha component)
        uncut:length of fMRI timeseries with edges
        max_shift: length of cut edges of fMRI timeseries
    """
    # plot normalized alpha regressor and data for region with lowest / highest abs correlation
    best_region = np.argmax(np.abs(corr))  # or random region?
    max_time = 100
    plt.plot(normalize_data(alpha[best_region, :max_time]), label=f"alpha {mode}")
    plt.plot(
        normalize_data(alpha_filt[best_region, :max_time]),
        label=f"alpha {mode} filt",
    )
    plt.plot(
        normalize_data(
            fMRI_timeseries[ex_participant][
                best_region,
                shift : uncut - (max_shift - shift),
            ][:max_time]
        ),
        label="fMRI",
    )
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("normalized values")
    plt.title(
        f"comparison of EEG {mode} and fMRI\nfor participant {ex_participant} and region {best_region}"
    )
    plt.show()


def alpha_mean_corrs(
    fMRI_timeseries, EEG_timeseries, regionsMap, HRF_resEEG, sampling_freq=200
):

    """
    For every particpant, the correlation between
    - alpha regressor and fMRI timeseries
    - alpha band power and fMRI timeseries
    is calculated within every region.
    The resulting matrix of correlation is averaged.
    Different shifts of the alpha component relative to the fMRI are considered, the one producing
    the highest absolute-lowest- mean for both comparisons is returned together with the corresponding mean.
    Plots alpha component vs fMRI timeseries for region with highest absolute correlation
    for chosen shift for every participant.
    arguments:
        fMRI_timeseries: fMRI timeseries (NOT interpolated)
        EEG_timeseries: EEG timeseries
        regionsMap: [68 x 1] vector that contains the region sorting of source
                    activity as outputted by Brainstorm
        HRF_resEEG: [1 x 6498] vector that contains the hemodynamic
                    response function sampled at 200 Hz
        sampling_freq: EEG sampling frequency
    returns:
        mean_reg_all: lowest mean of correlations for alpha regressors for each partiicpant
        mean_power_all: lowest mean of correlations for alpha band power for each partiicpant
        shifts_reg: corresponding shifts for mean_reg_all
        shifts_power: corresponding shifts for mean_reg_all
    """

    N = len(fMRI_timeseries)
    shifts_reg = np.zeros((N), dtype=int)
    shifts_power = np.zeros((N), dtype=int)
    mean_reg_all = np.empty((N))
    mean_power_all = np.empty((N))

    for participant in np.arange(N):
        fMRI_curr = fMRI_timeseries[participant]

        # with HRF convolution:
        alpha_reg, alpha_reg_filt = compute_alpha_regressor(
            EEG_timeseries, participant, regionsMap, HRF_resEEG, sampling_freq
        )

        # without HRF convolution but cut edges:
        alpha_power, alpha_power_filt = compute_power_timeseries(
            EEG_timeseries, participant, regionsMap, sampling_freq
        )

        uncut = fMRI_curr.shape[1]
        cut = alpha_reg.shape[1]

        max_shift = uncut - cut

        corr_reg_final = np.empty((68))
        corr_power_final = np.empty((68))
        mean_reg_all[participant] = 0
        mean_power_all[participant] = 0
        for s in np.arange(max_shift):

            # ? compare average corr --> keep smallest (bs negative corr expected)?
            # ! alternative: keep the one with the largest absolute value?

            # for alpha regressor: only keep correlations within same regions
            corr_reg = np.diagonal(
                corrcoef2D(fMRI_curr[:, s : uncut - (max_shift - s)], alpha_reg_filt)
            )
            # ? take mean over all region correlations
            mean_reg = np.mean(corr_reg)

            if np.abs(mean_reg) > np.abs(mean_reg_all[participant]):
                mean_reg_all[participant] = mean_reg
                corr_reg_final = corr_reg
                shifts_reg[participant] = s

            # repeat for alpha power band
            corr_power = np.diagonal(
                corrcoef2D(fMRI_curr[:, s : uncut - (max_shift - s)], alpha_power_filt)
            )
            mean_power = np.mean(corr_power)

            if np.abs(mean_power) > np.abs(mean_power_all[participant]):
                mean_power_all[participant] = mean_power
                corr_power_final = corr_power
                shifts_power[participant] = s

        # plot alpha reg/power and fMRI in region with highest abs correlation
        plot_alpha_reg_power_fMRI(
            alpha_reg,
            alpha_reg_filt,
            fMRI_timeseries,
            corr_reg_final,
            "reg",
            participant,
            shifts_reg[participant],
            uncut,
            max_shift,
        )
        plot_alpha_reg_power_fMRI(
            alpha_power,
            alpha_power_filt,
            fMRI_timeseries,
            corr_power_final,
            "power",
            participant,
            shifts_power[participant],
            uncut,
            max_shift,
        )

        print(
            f"participant {participant+1}\nhighest abs corr over time shifts for alpha regressor:\n{mean_reg_all[participant]}\nhighest abs corr over time shifts for alpha power band:\n{mean_power_all[participant]}"
        )
    return (
        alpha_reg,
        alpha_reg_filt,
        alpha_power,
        alpha_power_filt,
        mean_reg_all,
        mean_power_all,
        shifts_reg,
        shifts_power,
    )


def plot_compare_alpha(mean_reg_all, mean_power_all, shifts_reg, shifts_power):
    """
    Plots scatter plots to compare lowest mean correlations between
    - alpha regressor and fMRI timeseries
    - alpha band power and fMRI timeseries
    and the corresponding shifts. The point colors show which datapoints belong to one participant.
    arguments:
        mean_reg_all: lowest mean of correlations for alpha regressors for each partiicpant
        mean_power_all: lowest mean of correlations for alpha band power for each partiicpant
        shifts_reg: corresponding shifts for mean_reg_all
        shifts_power: corresponding shifts for mean_reg_all
    """
    N = len(mean_reg_all)
    cm = mcm.get_cmap("RdYlBu")
    colors = [cm(1.0 * i / N) for i in range(N)]

    plt.scatter(mean_reg_all, mean_power_all, c=colors)
    plt.xlabel("alpha regressor")
    plt.ylabel("alpha band power")
    plt.title(
        "comparison of mean correlations between fMRI and\nEEG alpha regressor vs band power for all participants"
    )
    plt.show()
    plt.scatter(shifts_reg, shifts_power, c=colors)
    plt.xlabel("alpha regressor")
    plt.ylabel("alpha band power")
    plt.title(
        "comparison of chosen shift for correlation between fMRI and\nEEG alpha regressor vs band power for all participants"
    )
    plt.show()


# %%
