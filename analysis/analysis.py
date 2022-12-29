# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as interp

from scipy import io as sio
from pygsp import graphs

from helpers import *

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
### next steps
# - compare timeseries EEG & fMRI (e.g. lower vs higehr half of harmonics)
# - Fischer TRansform instead of mean for correlations
# - scale timeseries before applying similarity measures
#### other ToDos
# - fix time axis in all plots
# - save plots
# - data in numpy arrays instead of lists
# - compare scipy.interpolate.interp1d & scipy.signal.resample
# - general refactoring

# %%


def _loop_participants(func):
    def wrapper(*args):
        if args[0].loop_participants:
            for participant in np.arange(args[0].N):
                args[0].ex_participant = participant
                func(*args)
        else:
            func(*args)

    return wrapper


def _loop_regions(func):
    def wrapper(*args):
        if args[0].loop_regions:
            for region in np.arange(args[0].N_regions):
                args[0].ex_region = region
                func(*args)
        else:
            func(*args)

    return wrapper


def _loop_domains(func):
    def wrapper(*args):
        for domain in args[0].domains:
            args[0].domain = domain
            func(*args)

    return wrapper


def _loop_modalities(func, interp=True):
    def wrapper(*args):
        for modality in args[0].modalities:
            args[0].modality = modality
            args[0].choose_modality_data(modality, interp)
            func(*args)

    return wrapper


class Data:
    """
    class for all data and methods for generation of overview plots and data analysis
    """

    def __init__(self, mode, loop_participants=True, loop_regions=True):

        self.mode = mode
        self.loop_participants = loop_participants
        self.loop_regions = loop_regions

        # paths
        # data
        self.SC_path = "../data/empirical_structural_connectomes/SCs.mat"
        self.fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
        self.EEG_data_path = "../data/empirical_source_activity/source_activity.mat"
        # EEG regions
        self.EEG_regions_path = "../data/empirical_source_activity/regionsMap.mat"
        # region coordinates
        self.coords_path = "../data/ROI_coords.mat"
        # HRF
        self.HRF_path = "../data/HRF_200Hz.mat"

        # known variables about data
        self.N = 15
        self.N_regions = 68
        self.fMRI_timesteps = 661
        self.EEG_timesteps = 259184
        self.sampling_freq = 200  # Hz, sampling frequency for EEG

        # other variable settings
        self.ex_participant = 1
        self.ex_harmonic = 5
        self.ex_region = 5
        self.domains = ["region", "harmonic"]
        self.domain = None
        self.modalities = ["EEG", "fMRI"]
        self.modality = None

        # initialize variables for data and IDs
        # structural connectivity matrices  graphs
        self.G = None
        self.Gs = []
        self.SC_weights = np.empty((self.N_regions, self.N_regions, self.N))
        self.mean_SC_weights = None
        self.SC_participant_IDs = []
        self.SC_region_IDs = []
        # EEG: (68x259184)x15 for (RegionsxTime)xparticipants
        self.EEG_timeseries = np.empty((self.N_regions, self.EEG_timesteps, self.N))
        self.trans_EEG_timeseries = np.empty(
            (self.N_regions, self.EEG_timesteps, self.N)
        )
        self.EEG_participant_IDs = []
        self.EEG_region_IDs = []
        # fMRI: before interpolation: (68x661)x15 for (RegionsxTime)xparticipants
        # after interpolation as in EEG: (68x259184)x15 for (RegionsxTime)xparticipants
        self.fMRI_timeseries = np.empty((self.N_regions, self.fMRI_timesteps, self.N))
        self.fMRI_timeseries_interp = np.empty(
            (self.N_regions, self.EEG_timesteps, self.N)
        )
        self.trans_fMRI_timeseries = np.empty(
            (self.N_regions, self.EEG_timesteps, self.N)
        )
        self.fMRI_participant_IDs = []
        self.fMRI_region_IDs = []
        # other
        self.coords = None
        self.HRF = sio.loadmat(self.HRF_path)["HRF_resEEG"]

        # for analysis methods
        self.low_harm = None
        self.high_harm = None
        self.timeseries = None
        self.trans_timeseries = None
        self.power = None
        self.regions_all_corrs = None
        self.harmonics_all_corrs = None
        self.mean_regions_corrs = None
        self.mean_harmonics_corrs = None
        # for sanity checks
        self.all_corrs_reg = None
        self.all_corrs_power = None
        self.mean_reg_all = None
        self.mean_power_all = None
        self.shifts_reg = None
        self.shifts_power = None
        self.max_shift = None

        # get data
        self._get_coords()
        self._get_SC_matrices()
        self._get_functional_data()

    def _get_coords(self):

        coords_all = sio.loadmat(self.coords_path)["ROI_coords"]
        coords_labels = sio.loadmat("../data/ROI_coords.mat")["all_labels"].flatten()
        self.roi_IDs = np.hstack(
            sio.loadmat("../data/empirical_fMRI/empirical_fMRI.mat")[
                "freesurfer_roi_IDs"
            ][0].flatten()
        ).astype("int32")
        idx_rois = np.empty((len(self.roi_IDs)), dtype="int32")
        for idx, roi in enumerate(self.roi_IDs):
            idx_rois[idx] = np.argwhere(coords_labels == roi)
        self.coords = coords_all[idx_rois]

    def _get_SC_matrices(self):
        unflattened_SC_data = sio.loadmat(self.SC_path)["SC"]
        SC_data = np.ndarray.flatten(unflattened_SC_data)

        for participant in np.arange(self.N):
            SC_weights_pre, SC_participant_ID_curr = SC_data[participant]
            # extract weights from nested object
            # make weights symmetric t keep going for now
            SC_weights_pre[0, 0][0] = (
                SC_weights_pre[0, 0][0] + SC_weights_pre[0, 0][0].T
            )
            self.SC_weights[:, :, participant] = SC_weights_pre[0, 0][0]
            # add participant ID
            self.SC_participant_IDs.append(SC_participant_ID_curr)

    def _get_functional_data(self):

        self.__get_raw_functional_data()
        self.__sort_EEG_data()

        # transform data
        if self.mode == "ind":
            self.__transform_w_ind_SCs()
        else:
            self.__transform_w_mean_SC()

        self.__check_participant_IDs()
        self.__check_region_IDs()

    def __get_raw_functional_data(self):

        # load fMRI data
        unflattened_fMRI_data = sio.loadmat(self.fMRI_data_path)["fMRI"]
        fMRI_data = np.ndarray.flatten(unflattened_fMRI_data)

        # load EEG data
        EEG_data_file = h5py.File(self.EEG_data_path, "r")

        for participant in np.arange(self.N):

            # get EEG participant IDs
            EEG_ID_store_curr = EEG_data_file["source_activity/sub_id"][participant][0]
            EEG_participant_ID = EEG_data_file[EEG_ID_store_curr][:]
            EEG_participant_ID = (
                "['" + "".join(chr(c[0]) for c in EEG_participant_ID) + "']"
            )
            self.EEG_participant_IDs.append(EEG_participant_ID)
            # get participant's EEG data
            EEG_data_store_curr = EEG_data_file["source_activity/ts"][participant][0]
            EEG_timeseries_curr = EEG_data_file[EEG_data_store_curr]
            # subtract mean over regions ?
            EEG_timeseries_curr = EEG_timeseries_curr - np.mean(EEG_timeseries_curr, 0)
            self.EEG_timeseries[:, :, participant] = EEG_timeseries_curr[:, :].T

            # get participant's fMRI data
            fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
            self.fMRI_timeseries[:, :, participant] = fMRI_timeseries_curr.T
            self.fMRI_participant_IDs.append(fMRI_participant_ID)
            # stretch fMRI data over time to EEG sequence length
            for region in np.arange(self.N_regions):
                fMRI_interp = interp.interp1d(
                    np.arange(self.fMRI_timesteps),
                    self.fMRI_timeseries[region, :, participant],
                )
                self.fMRI_timeseries_interp[region, :, participant] = fMRI_interp(
                    np.linspace(0, self.fMRI_timesteps - 1, self.EEG_timesteps)
                )
        # plot interpolated fMRI timeseries
        plot_ex_interp(
            self.fMRI_timeseries,
            self.fMRI_timeseries_interp,
            self.ex_participant,
            self.ex_harmonic,
        )

        # variables for analysis
        self.EEG_power = None
        self.fMRI_power = None

    def __sort_EEG_data(self):

        """
        EEG data is known to have different region sorting than (f)MRI data
        """
        self.SC_region_IDs = np.hstack(
            sio.loadmat(self.SC_path)["freesurfer_roi_IDs"][0].flatten()
        ).astype("uint16")

        self.EEG_region_IDs = sio.loadmat(self.EEG_regions_path)["regionsMap"][:, 0]

        self.fMRI_region_IDs = np.hstack(
            sio.loadmat(self.fMRI_data_path)["freesurfer_roi_IDs"][0].flatten()
        ).astype("uint16")

        region_sort_EEG_after_SC = np.empty((self.N_regions), dtype=int)
        for region in np.arange(self.N_regions):
            region_sort_EEG_after_SC[region] = np.argwhere(
                self.EEG_region_IDs == self.SC_region_IDs[region]
            )

        self.EEG_timeseries = self.EEG_timeseries[region_sort_EEG_after_SC, :, :]
        self.EEG_region_IDs = self.EEG_region_IDs[region_sort_EEG_after_SC]

    def __transform_w_mean_SC(self):

        self.mean_SC_weights = np.mean(self.SC_weights, 2)

        # compute one graph for all participants
        self.G = graphs.Graph(
            self.mean_SC_weights, lap_type="normalized", coords=self.coords
        )
        self.G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e

        # get spectral representation of signal
        for participant in np.arange(self.N):
            self.trans_EEG_timeseries[:, :, participant] = self.G.gft(
                self.EEG_timeseries[:, :, participant]
            )
            self.trans_fMRI_timeseries[:, :, participant] = self.G.gft(
                self.fMRI_timeseries_interp[:, :, participant]
            )

    def __transform_w_ind_SCs(self):

        for participant in np.arange(self.N):

            # compute individual graph for each participant
            G = graphs.Graph(
                self.SC_weights[:, :, participant],
                lap_type="normalized",
                coords=self.coords,
            )
            G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e
            self.Gs.append(G)

            # get spectral representation of signal
            self.trans_EEG_timeseries[:, :, participant] = self.Gs[participant].gft(
                self.EEG_timeseries[:, :, participant]
            )
            self.trans_fMRI_timeseries[:, :, participant] = self.Gs[participant].gft(
                self.fMRI_timeseries_interp[:, :, participant]
            )

    def __check_participant_IDs(self):
        ID_count = 0
        for participant in np.arange(self.N):
            if (
                str(self.SC_participant_IDs[participant])
                == str(self.EEG_participant_IDs[participant])
            ) and (
                str(self.SC_participant_IDs[participant])
                == str(self.fMRI_participant_IDs[participant])
            ):
                ID_count += 1

        if ID_count == self.N:
            print(
                "all participant IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
            )

    def __check_region_IDs(self):
        ID_count = 0
        for region in np.arange(self.N_regions):
            if (
                str(self.SC_region_IDs[region]) == str(self.EEG_region_IDs[region])
            ) and (
                str(self.SC_region_IDs[region]) == str(self.fMRI_region_IDs[region])
            ):
                ID_count += 1

        if ID_count == self.N_regions:
            print(
                "all region IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
            )

    # _______
    # analysis methods
    def choose_modality_data(self, modality, interp):
        """
        helper function for loop modality decorator
        """
        if not ((self.low_harm != None) & (self.high_harm != None)):
            self.get_power()
        if modality == self.modalities[0]:
            self.timeseries = self.EEG_timeseries
            self.trans_timeseries = self.trans_EEG_timeseries
            self.power = self.EEG_power
        else:
            if interp:
                self.timeseries = self.fMRI_timeseries_interp
                self.trans_timeseries = self.trans_fMRI_timeseries
            else:
                self.timeseries = self.fMRI_timeseries
                self.trans_timeseries = self.trans_fMRI_timeseries
            self.power = self.fMRI_power

    def get_power(self, low_harm=0, high_harm=68):
        """
        returns power of transformed signal normalized for every timestep (all participant, all harmonics)
        """

        # normalize
        # mean over wrong axis?
        if (low_harm == 0) and (high_harm == self.N_regions):
            EEG_power = self.trans_EEG_timeseries[low_harm:high_harm, :, :] ** 2
            fMRI_power = self.trans_fMRI_timeseries[low_harm:high_harm, :, :] ** 2
            self.EEG_power = EEG_power / np.sum(EEG_power, 0)[np.newaxis, :]
            self.fMRI_power = fMRI_power / np.sum(fMRI_power, 0)[np.newaxis, :]
        else:
            if self.modality == self.modalities[0]:
                power = self.trans_EEG_timeseries[low_harm:high_harm, :, :] ** 2
            else:
                power = self.trans_fMRI_timeseries[low_harm:high_harm, :, :] ** 2
            return power / np.sum(power, 0)[np.newaxis, :]

        self.low_harm = low_harm
        self.high_harm = high_harm

    @_loop_participants
    @_loop_domains
    def plot_signal(self):
        plot_ex_signal_EEG_fMRI(
            self.EEG_timeseries,
            self.fMRI_timeseries_interp,
            self.ex_participant,
            self.domain,
        )

    @_loop_participants
    @_loop_regions
    @_loop_domains
    def plot_signal_single_domain(self):
        plot_ex_signal_fMRI_EEG_one(
            self.EEG_timeseries,
            self.fMRI_timeseries_interp,
            self.ex_participant,
            self.ex_region,
            self.domain,
        )

    @_loop_participants
    @_loop_modalities
    def plot_domain(self):
        plot_ex_regions_harmonics(
            self.timeseries, self.trans_timeseries, self.ex_participant, self.modality
        )

    @_loop_participants
    @_loop_modalities
    def plot_power_stem_cum(self, low_harm=0, high_harm=68):
        """
        Plot power per harmonic and cumulative power for individual participant(s)
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        plot_power_stem(self.power, self.modality, self.ex_participant)
        plot_power_cum(self.power, self.modality, self.ex_participant)

    @_loop_modalities
    def plot_power_mean_stem_cum(self, low_harm=0, high_harm=68):
        """
        EEG sanity check 1: EEG power, mean over participants
        Plot power per harmonic and cumulative power over all participants
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        plot_power_stem(self.power, self.modality)
        plot_power_cum(self.power, self.modality)

    @_loop_participants
    def plot_power_corr(self, low_harm=0, high_harm=68):
        """
        Plot power over time and power correlation for fMRI and EEG for individual participant(s)
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        # plot fMRI-EEG power in comparison and then correlation
        plot_ex_power_EEG_fMRI(self.EEG_power, self.fMRI_power, self.ex_participant)
        plot_power_corr(self.EEG_power, self.fMRI_power, self.ex_participant)

    def plot_power_mean_corr(self, low_harm=0, high_harm=68):
        """
        Plot power over time and power correlation for fMRI and EEG over all participants
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        # plot fMRI-EEG power in comparison and then correlation
        plot_ex_power_EEG_fMRI(self.EEG_power, self.fMRI_power)
        plot_power_corr(self.EEG_power, self.fMRI_power)

    def plot_EEG_freq_band(self):
        """
        EEG sanity check 2: plot EEG frequency band
        """
        # use all EEG data
        freqs, psd = plot_fft_welch(
            np.moveaxis(self.EEG_timeseries, -1, 0).flatten(), self.sampling_freq
        )

    # _______
    # EEG sanity check 3: EEG-fMRI -> see helpers/sanity_check.py
    def get_alpha_corrs(self):
        """
        Computes correlations for alpha band EEG and fMRI and plots comparison for best shift&region per participant
        """
        (
            self.all_corrs_reg,
            self.all_corrs_power,
            self.mean_reg_all,
            self.mean_power_all,
            self.shifts_reg,
            self.shifts_power,
            self.max_shift,
        ) = alpha_mean_corrs(self.fMRI_timeseries, self.EEG_timeseries, self.HRF)
        # plot compare alpha to compare (lowest) mean correlations

    def plot_alpha_corrs(self):
        """
        Plots correlations for alpha band EEG and fMRI over participants and regions
        """
        plot_alpha_corr(self.all_corrs_reg, self.all_corrs_power, self.max_shift)

    # _______
    def get_vertex_vs_graph(self):
        # compare vertex vs graph domain: correlation between EEG & fMRI
        (
            self.regions_all_corrs,
            self.harmonics_all_corrs,
            self.mean_regions_corrs,
            self.mean_harmonics_corrs,
            ttest_results,
        ) = vertex_vs_graph(
            self.EEG_timeseries,
            self.fMRI_timeseries_interp,
            self.trans_EEG_timeseries,
            self.trans_fMRI_timeseries,
        )
        return ttest_results

    def plot_vertex_vs_graph(self):
        plot_vertex_vs_graph_corr(self.regions_all_corrs, self.harmonics_all_corrs)

    @_loop_modalities
    def get_lower_vs_upper_harmonics(self):
        """
        compare lower vs upper half of harmonics
        """
        half = int(self.N_regions / 2)
        power_low = self.get_power(0, half)
        power_high = self.get_power(half, self.N_regions)
        # plots in subplots? / plot 1 plot with vertical line at half?
        plot_power_stem(power_low, self.modality)
        plot_power_stem(power_high, self.modality, start=half)

        ttest_greater(
            power_low,
            power_high,
            [
                ["low", "high"],
                "power",
                f"power between lower and higher {self.modality} harmonics",
            ],
        )

    def get_GE(self):
        simi_betw_participants(self.Gs, simi_GE, "GE", self.N)

    @_loop_modalities
    def get_TVG(self):
        simi_betw_participants(
            self.Gs, simi_TVG, "TVG", self.N, self.modality, self.trans_timeseries
        )

    @_loop_modalities
    def get_JET(self):
        simi_betw_participants(
            self.Gs, simi_JET, "JET", self.N, self.modality, self.trans_timeseries
        )


# %%
mode = "mean"  # 'mean' or 'ind'
# data_mean = data('mean')
data_ind = Data(mode="ind", loop_participants=False)

# %%
# data_ind.plot_signal()
# data_ind.plot_signal_single_domain()
# data_ind.plot_domain()
# data_ind.plot_power_stem_cum()
# data_ind.plot_power_mean_stem_cum()
# data_ind.plot_power_corr()
# data_ind.plot_power_mean_corr()
# data_ind.plot_EEG_freq_band()
# data_ind.get_alpha_corrs()
# data_ind.plot_alpha_corrs()
# data_ind.get_vertex_vs_graph()
# data_ind.plot_vertex_vs_graph()
# data_ind.get_lower_vs_upper_harmonics()
data_ind.get_GE()
data_ind.get_TVG()
data_ind.get_JET()


# %%
# ____________________________
# everything below has to be rewritten for data class


# plot signal on graph
plot_ex_graphs_3D(Gs, trans_EEG_timeseries, ex_participant, "EEG")
plot_ex_graphs_3D(Gs, trans_fMRI_timeseries, ex_participant, "fMRI")


# plot eigenvectors on graphs

N_plots = 3

fig, axes = plt.subplots(1, N_plots, figsize=(10, 3), subplot_kw=dict(projection="3d"))
for t, ax in enumerate(axes):
    G.plot_signal(
        G.U[:, t],
        vertex_size=30,
        show_edges=True,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"harmonic {t + 1}")
    ax.axis("off")
    plt.suptitle(f"first {N_plots} harmonics")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, N_plots, figsize=(10, 3), subplot_kw=dict(projection="3d"))
for t, ax in enumerate(axes):
    G.plot_signal(
        G.U[:, -(t + 1)],
        vertex_size=30,
        show_edges=True,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"harmonic {68 -t}")
    ax.axis("off")
    plt.suptitle(f"last {N_plots} harmonics")
fig.tight_layout()
plt.show()


# %%
fig, axes = plt.subplots(1, N_plots, figsize=(10, 3), subplot_kw=dict(projection="3d"))
for t, ax in enumerate(axes):
    G.plot_signal(
        G.U[:, t + 30],
        vertex_size=30,
        show_edges=True,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"harmonic {t+31}")
    ax.axis("off")
    plt.suptitle(f"last {N_plots} harmonics")
fig.tight_layout()
plt.show()
