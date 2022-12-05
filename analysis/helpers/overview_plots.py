import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import signal as sg


def corrcoef2D(A, B):
    """
    from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
    """
    example:
    a = np.array(([1, 3, 7], [1, 2, 7]))
    b = np.array(([1, 3, 5], [1, 3, 7]))

    print(f"a:\n{a}\nb:\n{b}")

    print(f"row-wise corr:\n{corrcoef2D(a, b)}")
    """


def plot_ex_interp(timeseries, timeseries_interp, ex_participant, ex_harmonic):
    """
    plots exemplary timeseries and interpolated timeseries for comparison (for 1 participant, 1 harmonic)
    arguments:
        timeseries: array before interpolation
        timeseries_interp: array after interpolation
        ex_participant: example participant index
        ex_harmonic: example harmonic index
    """
    plt.subplot(211)
    plt.plot(timeseries[ex_participant][ex_harmonic, :])
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.title("original fMRI signal")
    plt.subplot(212)
    plt.plot(timeseries_interp[ex_participant][ex_harmonic, :])
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.title("stretched fMRI signal")
    plt.tight_layout()
    plt.show()


def plot_ex_signal_EEG_fMRI(EEG_timeseries, fMRI_timeseries, ex_participant, mode):
    """
    plots exemplary signal of EEG and fMRI signal over time (for 1 participant, 5 regions/harmonics)
    arguments:
        EEG_timeseries: EEG timeseries
        fMRI_timeseries: fMRI timeseries (same length as EEG timeseries)
        ex_participant: example participant index
        mode: string, should be 'region' or 'harmonic'
    """
    N_regions, _ = EEG_timeseries[ex_participant].shape

    signal = "signal"
    if mode == "region":
        signal = "activity [a.u.]"
    elif mode == "harmonic":
        signal = "GFT weights"

    # normalize each line?
    plt.subplot(211)
    for i in np.linspace(0, N_regions - 1, 5).astype(int):
        plt.plot(EEG_timeseries[ex_participant][i, :], label=f"{mode} {i+1}")
    plt.xlabel("time")
    plt.ylabel(signal)
    plt.title("examplary EEG signal for one participant")
    plt.legend()
    plt.subplot(212)
    for i in np.linspace(0, N_regions - 1, 5).astype(int):
        plt.plot(fMRI_timeseries[ex_participant][i, :], label=f"{mode} {i+1}")
    plt.xlabel("time")
    plt.ylabel(signal)
    plt.title("examplary fMRI signal for one participant")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ex_signal_fMRI_EEG_one(
    EEG_timeseries, fMRI_timeseries, ex_participant, ex_region_harmonic, mode
):
    """
    plots exemplary timeseries of EEG and fMRI signal for comparison (for 1 participant, 1 harmonic)
    arguments:
        EEG_timeseries: EEG timeseries
        fMRI_timeseries: fMRI timeseries interpolated (same length as EEG timeseries)
        ex_participant: example participant index
        ex_region_harmonic: example region or harmonic index
        mode: string, should be 'region' or 'harmonic'
    """

    _, EEG_timesteps = EEG_timeseries[ex_participant].shape

    signal = "signal"
    if mode == "region":
        signal = "activity [a.u.]"
    elif mode == "harmonic":
        signal = "GFT weights"

    plt.plot(
        fMRI_timeseries[ex_participant][ex_region_harmonic, :]
        / np.max(np.abs(fMRI_timeseries[ex_participant][ex_region_harmonic, :])),
        label="fMRI",
        alpha=0.7,
    )
    plt.plot(
        EEG_timeseries[ex_participant][ex_region_harmonic, 100 : EEG_timesteps - 101]
        / np.max(
            np.abs(
                EEG_timeseries[ex_participant][
                    ex_region_harmonic, 100 : EEG_timesteps - 101
                ]
            )
        ),
        label="EEG",
        alpha=0.7,
    )
    plt.legend()
    plt.xlabel("time")
    plt.ylabel(f"scaled {signal}")
    plt.title(
        f"exemplary comparison of {mode} {ex_region_harmonic + 1} in EEG and fMRI\n for participant {ex_participant + 1}"
    )
    plt.show()


def plot_ex_regions_harmonics(timeseries, trans_timeseries, ex_participant, mode):
    """
    plots exemplary activity and GFT weights over time (for 1 participant, all regions/harmonics)
    arguments:
        timeseries: activity matrix
        trans_timeseries: GFT weight matrix
        ex_participant: example participant index
        mode: string, should be 'EEG' or 'fMRI'
    """

    # 10**5 used in Glomb et al. (2020) for same plots ?
    factor = 10**5
    activity_scaled = timeseries[ex_participant] * factor
    weights_scaled = trans_timeseries[ex_participant] * factor

    # EEG activity in original domain
    plt.subplot(211)
    map = sns.heatmap(
        activity_scaled / np.max(activity_scaled),
        cbar_kws={"label": "activity $* 10^5$ [a.u.]"},
    )
    map.set_xlabel("time", fontsize=10)
    map.set_ylabel("brain region", fontsize=10)
    plt.title(
        f"{mode} activity for all brain areas over time\n for participant {ex_participant+1}"
    )
    plt.subplot(212)
    # EEG activity in graph frequency/spectral domain
    map = sns.heatmap(
        weights_scaled / np.max(weights_scaled),
        cbar_kws={"label": "GFT weights $* 10^5$ [a.u.]"},
    )
    map.set_xlabel("time", fontsize=10)
    map.set_ylabel("network harmonic", fontsize=10)
    plt.title(
        f"{mode} activity for all network harmonics over time\n for participant {ex_participant+1}"
    )
    plt.tight_layout()
    plt.show()


def ex_EEG_fMRI_corr(EEG_timeseries, fMRI_timeseries, ex_participant, mode):
    """
    plots correlation between EEG and fMRI signal (1 participant, all regions/harmonics)
    arguments:
        EEG_timeseries: EEG timeseries
        fMRI_timeseries: fMRI timeseries (same length as EEG timeseries)
        ex_participant: example participant index
        mode: string, should be 'region' or 'harmonic'
    return:
        fMRI_EEG_corr: correlation matrix of correlation between fMRI & EEGH regions/harmonics
    """
    fMRI_EEG_corr = corrcoef2D(
        fMRI_timeseries[ex_participant], EEG_timeseries[ex_participant]
    )
    map = sns.heatmap(fMRI_EEG_corr)
    map.set_xlabel("EEG ?", fontsize=10)
    map.set_ylabel("fMRI ?", fontsize=10)
    plt.title(
        f"correlation of {mode}s in fMRI and EEG for participant {ex_participant+1}"
    )
    plt.show()
    return fMRI_EEG_corr


def power_mean(signal, ex_participant, mode):
    """
    plots mean power over time (1 participant, all harmonics)
    arguments:
        signal: GFT weights
        ex_participant: example participant index
        mode: string, should be 'EEG' or 'fMRI'
    return:
        power: power per harmonic
    """
    # mean power (L2 norm) over time
    # or sqrt of L2 norm??
    # square in right place?
    # does this make sense with a mean over time? -> analogous to EEG/fMRI power plots above, otherwise timesteps instead of harmonics are important
    # normalize power vector to 1 --> normalize power to 1 at every point in time????
    # normalize power at every time point? and then also divide by number of regions?
    power = signal[ex_participant] ** 2
    power = np.mean(power / np.sum(power, 0)[np.newaxis, :], 1)

    plt.stem(power)
    plt.xlabel("harmonic")
    plt.ylabel("signal strength")
    plt.title(
        f"{mode} normalized graph frequency domain for participant {ex_participant + 1}\n (mean over time, normalized also per timepoint)"
    )
    plt.show()
    return power


def plot_cum_power(power_mean, ex_participant, mode):
    """
    plots cumulative power (mean over time) (1 participant, all harmonics)
    arguments:
        power_mean: mean power over time
        ex_participant: example participant index
        mode: string, should be 'EEG' or 'fMRI'
    """
    N_regions = power_mean.size
    for i in np.arange(50):
        curr_random = np.random.uniform(0, 1, N_regions)
        plt.plot(np.cumsum(curr_random) / np.sum(curr_random), color="grey", alpha=0.1)
    plt.plot(np.cumsum(power_mean), label="SC graph")
    plt.xlabel("harmonic")
    plt.ylabel("cumulative power ?")
    plt.title(
        f"{mode} power captured cumulatively for participant {ex_participant + 1}\n (mean over time, normalized also per timepoint)"
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label="random graphs", color="grey", alpha=0.1)
    handles.extend(
        [
            line,
        ]
    )
    plt.legend(handles=handles)
    plt.show()
    """
    # normalize power only after taking the mean, not per time point?
    t_mean_freq = np.mean(trans_EEG_timeseries[ex_participant] ** 2, 1)
    t_mean_freq_norm = t_mean_freq / np.sum(t_mean_freq)
    plt.stem(t_mean_freq_norm)
    plt.xlabel("harmonic")
    plt.ylabel("signal strength")
    plt.title(
        f"normalized graph frequency domain for participant {ex_participant + 1} (mean over time)"
    )
    plt.show()
    for i in np.arange(50):
        curr_random = np.random.uniform(0, 1, N_regions)
        plt.plot(np.cumsum(curr_random) / np.sum(curr_random), color="grey", alpha=0.1)
    plt.plot(np.cumsum(t_mean_freq_norm), label="SC graph")
    plt.xlabel("harmonic")
    plt.ylabel("cumulative power ?")
    plt.title(
        f"Power captured cumulatively for participant {ex_participant + 1} (mean over time)"
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label="random graphs", color="grey", alpha=0.1)
    handles.extend(
        [
            line,
        ]
    )
    plt.legend(handles=handles)
    plt.show()
    """


def power_norm(trans_timeseries, ex_participant):
    """
    returns power of transformed signal normalized for every timestep (1 participant, all harmonics)
    arguments:
        trans_timeseries: GFT weight matrix
        ex_participant: example participant index
    returns:
        power_norm: normalized power matrix
    """
    power = trans_timeseries[ex_participant] ** 2
    # power_norm = power / np.sum(power)
    # normalize power in every timestep instead of overall
    power_norm = power / np.sum(power, 0)
    return power_norm


def plot_ex_power_EEG_fMRI(EEG_power_norm, fMRI_power_norm, ex_participant):
    """
    plots exemplary power for EEG and fMRI time (for 1 participant, all harmonics)
    arguments:
        EEG_power_norm: EEG power matrix
        fMRI_power_norm: fMRI power matrix
        ex_participant: example participant index
    """

    plt.subplot(211)
    # EEG power
    map = sns.heatmap(
        EEG_power_norm,
        cbar_kws={"label": "EEG L2$^2$"},
    )
    map.set_xlabel("time", fontsize=10)
    map.set_ylabel("network harmonic", fontsize=10)
    plt.title(
        f"EEG power for all network harmonics over time\n for participant {ex_participant+1}"
    )
    plt.subplot(212)
    # fMRI power
    map = sns.heatmap(
        fMRI_power_norm,
        cbar_kws={"label": "fMRI L2$^2$"},
    )
    map.set_xlabel("time", fontsize=10)
    map.set_ylabel("network harmonic", fontsize=10)
    plt.title(
        f"fMRI power for all network harmonics over time\n for participant {ex_participant+1}"
    )
    plt.tight_layout()
    plt.show()
    print(
        "for lower plot, there should be a difference between top and bottom network harmonic activations ?"
    )


# integrate with other corr plot fct?
def plot_power_corr(EEG_power_norm, fMRI_power_norm, ex_participant):
    """
    plots exemplary correlation of EEG and FMRI power (for 1 participant, all harmonics)
    arguments:
        EEG_power_norm: EEG power matrix
        fMRI_power_norm: fMRI power matrix
        ex_participant: example participant index
    """
    map = sns.heatmap(corrcoef2D(EEG_power_norm, fMRI_power_norm))
    map.set_xlabel("EEG ?", fontsize=10)
    map.set_ylabel("fMRI ?", fontsize=10)
    plt.title(
        f"correlation of harmonics in fMRI and EEG for participant {ex_participant+1}"
    )
    plt.show()


def plot_fft_welch(signal, sampling_freq):
    """
    adapted from https://raphaelvallat.com/bandpower.html
    plot normalized power spectral density (Welch method)
    arguments:
        signal: signal data, 1 dimensional
        sampling_freq: sampling frequency [Hz]
    returns:
        freqs: Array of sample frequencies
        psd: power spectral density of signal
    """
    sampling_freq = 200  # Hz
    time = np.arange(signal.size) / sampling_freq

    # Define window length (4 seconds)
    win = 2 * sampling_freq
    freqs, psd = sg.welch(signal, sampling_freq, nperseg=win)
    psd = psd / np.linalg.norm(psd)
    plt.plot(freqs[freqs != 0], 1 / freqs[freqs != 0], label="1/f")  # scaling?
    # plt.semilogy(freqs, psd)
    plt.plot(freqs, psd)  # scaling?
    plt.xlabel("frequency [Hz]")
    plt.ylabel("normalized power spectral density [a.u./Hz ?]")
    # plt.ylim([0, psd.max() * 1.1])
    plt.xlim([1, 60])  # cutoff where high-/lowpass filters were applied
    plt.title("Welch's periodogram")
    plt.legend()
    plt.show()
    return freqs, psd


# plot brain surfaces
def plot_ex_graphs_3D(Gs, trans_timeseries, ex_participant, mode):
    """
    Plots signal on graph (1 participant, all harmonics, 1 timepoint).
    arguments:
        Gs: list of graphs (pygsp)
        trans_timeseries: list of GFT weight matrices
        ex_participant: example participant index
        mode: string, should be 'EEG' or 'fMRI'
    """
    timesteps = len(trans_timeseries[ex_participant].T)
    N_plots = 3
    stepsize = int(timesteps / N_plots)

    fig, axes = plt.subplots(
        1, N_plots, figsize=(10, 3), subplot_kw=dict(projection="3d")
    )
    for t, ax in enumerate(axes):
        Gs[ex_participant].plot_signal(
            trans_timeseries[ex_participant][:, stepsize * t],
            vertex_size=30,
            show_edges=True,
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"timepoint {stepsize * t + 1}")
        ax.axis("off")
        plt.suptitle(f"{mode}: participant {ex_participant}")
    fig.tight_layout()
