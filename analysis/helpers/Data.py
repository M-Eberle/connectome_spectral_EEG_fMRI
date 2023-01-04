import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as interp

from scipy import io as sio
from pygsp import graphs

from helpers.methods import *


class Data:

    """
    Class for all data and methods for generation of overview plots and data analysis.
    """

    def __init__(self, SC_mode, loop_participants=True, loop_regions=True):

        self.mode = SC_mode
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
        self.EEG_power = None
        self.fMRI_power = None
        self.power = None
        self.regions_all_corrs = None
        self.harmonics_all_corrs = None
        self.mean_regions_corrs = None
        self.mean_harmonics_corrs = None
        self.GE = None
        self.TVG = None
        self.JET = None
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
        """
        Load coordinates for brain atlas (used for plotting on graph).
        """

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
        """
        Load structural connectivity matrices for all participants.
        """
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
        """
        Load EEG and fMRI timeseries and transform onto graph according to structural connectivity (SC) matrices
        for all participants.
        Transformation onto graph depends on decision for individual SC matrices for each participant
        or one mean SC matrix.
        """

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
        """
        Helper function that loads EEG and fMRI timeseries. fMRI timeseries are interpolated
        to the resolution of the EEG timeseries. Plots examplary comparison between
        fMRI timeseries before and after interpolation.
        """
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

    def __sort_EEG_data(self):
        """
        EEG data has different region sorting than (f)MRI data. The timeseries
        and region_IDs are sorted according to the (f)MRI order.
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
        """
        Helper function for transformation of timeseries for mean SC matrix.
        """

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
        """
        Helper function for transformation of timeseries for individual SC matrices.
        """

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
        """
        Checks that participant order aligns for SC matrices, fMRI and EEG data.
        """
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
        """
        Checks that region order aligns for SC matrices, fMRI and EEG data.
        """
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
        Helper function for loop_modalities decorator
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
        Returns power of transformed signal normalized for every timestep (all participant, all harmonics).
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

    @loop_participants
    @loop_domains
    def plot_signal(self):
        """
        Plots signal for one participant, all regions/harmonics.
        """
        plot_ex_signal_EEG_fMRI(
            self.EEG_timeseries,
            self.fMRI_timeseries_interp,
            self.ex_participant,
            self.domain,
        )

    @loop_participants
    @loop_regions
    @loop_domains
    def plot_signal_single_domain(self):
        """
        Plots signal for one participant and one region/harmonic.
        """
        plot_ex_signal_fMRI_EEG_one(
            self.EEG_timeseries,
            self.fMRI_timeseries_interp,
            self.ex_participant,
            self.ex_region,
            self.domain,
        )

    @loop_participants
    @loop_modalities
    def plot_domain(self):
        """
        Plots exemplary activity and GFT weights over time.
        """
        plot_ex_regions_harmonics(
            self.timeseries, self.trans_timeseries, self.ex_participant, self.modality
        )

    @loop_participants
    @loop_modalities
    def plot_power_stem_cum(self, low_harm=0, high_harm=68):
        """
        Plots power per harmonic and cumulative power for individual participant(s)
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        plot_power_stem(self.power, self.modality, self.ex_participant)
        plot_power_cum(self.power, self.modality, self.ex_participant)

    @loop_modalities
    def plot_power_mean_stem_cum(self, low_harm=0, high_harm=68):
        """
        EEG sanity check 1
        Plots power per harmonic and cumulative power over all participants
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        plot_power_stem(self.power, self.modality)
        plot_power_cum(self.power, self.modality)

    @loop_participants
    def plot_power_corr(self, low_harm=0, high_harm=68):
        """
        Plots power over time and power correlation for fMRI and EEG for individual participant(s).
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        # plot fMRI-EEG power in comparison and then correlation
        plot_ex_power_EEG_fMRI(self.EEG_power, self.fMRI_power, self.ex_participant)
        plot_power_corr(self.EEG_power, self.fMRI_power, self.ex_participant)

    def plot_power_mean_corr(self, low_harm=0, high_harm=68):
        """
        Plots power over time and power correlation for fMRI and EEG over all participants.
        """
        if not ((self.low_harm == low_harm) & (self.high_harm == high_harm)):
            self.get_power()
        # plot fMRI-EEG power in comparison and then correlation
        plot_ex_power_EEG_fMRI(self.EEG_power, self.fMRI_power)
        plot_power_corr(self.EEG_power, self.fMRI_power)

    def plot_EEG_freq_band(self):
        """
        EEG sanity check 2:
        Plots EEG frequency band.
        """
        # use all EEG data
        freqs, psd = plot_fft_welch(
            np.moveaxis(self.EEG_timeseries, -1, 0).flatten(), self.sampling_freq
        )

    # _______
    # EEG sanity check 3: EEG-fMRI -> see helpers/sanity_check.py
    def get_alpha_corrs(self):
        """
        Computes correlations for alpha band EEG and fMRI and plots comparison for best shift&region per participant.
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
        Plots correlations for alpha band EEG and fMRI over participants and regions.
        """
        plot_alpha_corr(self.all_corrs_reg, self.all_corrs_power, self.max_shift)

    @loop_participants
    def plot_alpha_corrs_on_graph(self):
        alpha_corrs_on_graph(
            self.Gs, self.all_corrs_reg, self.all_corrs_power, self.ex_participant
        )

    # _______
    def get_vertex_vs_graph(self):
        """
        Compares vertex vs graph domain: correlation between EEG & fMRI.
        returns:
            ttest_results: result of ttests for correlations between EEG & fMRI in vertex vs graph domain
        """
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
        """
        Plots correlation between EEG and fMRI in vertex (regions) and graph.
        """
        plot_vertex_vs_graph_corr(self.regions_all_corrs, self.harmonics_all_corrs)

    @loop_modalities
    def get_lower_vs_upper_harmonics(self):
        """
        Compares lower vs upper half of harmonics.
        returns:
            ttest_results: result of ttests for lower vs upper half of harmonics
        """
        half = int(self.N_regions / 2)
        power_low = self.get_power(0, half)
        power_high = self.get_power(half, self.N_regions)
        # plots in subplots? / plot 1 plot with vertical line at half?
        plot_power_stem(power_low, self.modality)
        plot_power_stem(power_high, self.modality, start=half)

        ttest_results = ttest_greater(
            power_low,
            power_high,
            [
                ["low", "high"],
                "power",
                f"power between lower and higher {self.modality} harmonics",
            ],
        )
        return ttest_results

    # _______________________
    # below: currently only for individual SCs

    def get_GE(self):
        self.GE = simi_betw_participants(self.Gs, simi_GE, "GE", self.N)

    @loop_modalities
    def get_TVG(self):
        self.TVG = simi_betw_participants(
            self.Gs, simi_TVG, "TVG", self.N, self.modality, self.trans_timeseries
        )

    def get_random_TVG(self, random_weights=True):
        """
        Calculate TVG with random signal only or random signal and random weights on graph.
        arguments:
            random_weights: if True, weights are also random, if false, only signal is random
        """
        if random_weights:
            random_TVG = TVG_random_signal_and_weights(
                self.coords, self.N_regions, self.EEG_timesteps, self.N
            )
        else:  # only random signal
            random_TVG = TVG_random_signal(
                self.Gs, self.N_regions, self.EEG_timesteps, self.N
            )
        return random_TVG

    @loop_participants
    def TVG_evecs(self):
        TVG_betw_evecs(self.Gs, self.N_regions, self.ex_participant)

    @loop_modalities
    def get_JET(self):
        self.JET = simi_betw_participants(
            self.Gs, simi_JET, "JET", self.N, self.modality, self.trans_timeseries
        )

    # 3D plots on graphs
    @loop_participants
    @loop_modalities
    def plot_signal_on_graph(self):
        plot_ex_graphs_3D(
            self.Gs, self.trans_timeseries, self.ex_participant, self.modality
        )

    @loop_participants
    def plot_evecs_on_graph(self):
        plot_ex_evecs_3D(self.Gs, self.ex_participant)
