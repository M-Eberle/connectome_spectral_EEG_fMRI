from scipy import io as sio
from pygsp import graphs
import numpy as np
import h5py
import scipy.interpolate as interp


def _transform_ind_SCs(self):
    """
    SC matrices, EEG and fMRI timeseries from ‘Hybrid Brain Model data’ in data folder
    are extracted for all participants. The EEG data is sorted according to the SC matrices & fMRI data.
    The fMRI data is interpolated (stretched) to fit the length of the EEG data.
    The EEG and interpolated fMRI timeseries are transformed by the SC graph's
    Laplacian's eigenvectors.
    arguments:
        SC_path: path to SC matrices file
        EEG_data_path: path to EEG data file
        fMRI_data_path: path to fMRI data file
        EEG_regions_path: path to file with EEG region sorting
        coords: coordinates for graph nodes
    returns:
        SC_weights: list with SC matrix for each participant
        EEG_timeseries: list with EEG activity for each participant
        trans_EEG_timeseries: list with EEG GFT weights for each participant
        fMRI_timeseries: list with fMRI activity for each participant
        fMRI_timeseries_interp: list with interpolated fMRI activity for each participant
        trans_fMRI_timeseries: list with fMRI GFT weights for each participant
    """

    for participant in np.arange(self.N):

        # compute participant's graph
        G = graphs.Graph(
            self.SC_weights[participant], lap_type="normalized", coords=self.coords
        )
        G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e
        self.Gs.append(G)

    # get spectral representation of signal
    # use individual graph for each participant
    for participant in np.arange(self.N):
        self.trans_EEG_timeseries[:, :, participant] = self.Gs[participant].gft(
            self.EEG_timeseries[:, :, participant]
        )
        self.trans_fMRI_timeseries[:, :, participant] = self.Gs[participant].gft(
            self.fMRI_timeseries_interp[:, :, participant]
        )


def _transform_w_mean_SC(self):
    """
    SC matrices, EEG and fMRI timeseries from ‘Hybrid Brain Model data’ in data folder
    are extracted for all participants. The EEG data is sorted according to the SC matrices & fMRI data.
    The fMRI data is interpolated (stretched) to fit the length of the EEG data.
    The EEG and interpolated fMRI timeseries are transformed by the SC graph's
    Laplacian's eigenvectors.
    arguments:
        SC_path: path to SC matrices file
        EEG_data_path: path to EEG data file
        fMRI_data_path: path to fMRI data file
        EEG_regions_path: path to file with EEG region sorting
        coords: coordinates for graph nodes
    returns:
        mean_SC_weights: mean SC matrix
        G: graph generated from SC matrix
        EEG_timeseries: list with EEG activity for each participant
        trans_EEG_timeseries: list with EEG GFT weights for each participant
        fMRI_timeseries: list with fMRI activity for each participant
        fMRI_timeseries_interp: list with interpolated fMRI activity for each participant
        trans_fMRI_timeseries: list with fMRI GFT weights for each participant
    """

    # compute one graph for all participants
    self.G = graphs.Graph(
        self.mean_SC_weights, lap_type="normalized", coords=self.coords
    )
    self.G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e

    # spectral representation of signal
    # use one graph for all participants
    for participant in np.arange(self.N):
        self.trans_EEG_timeseries[:, :, participant] = self.G.gft(
            self.EEG_timeseries[:, :, participant]
        )
        self.trans_fMRI_timeseries[:, :, participant] = self.G.gft(
            self.fMRI_timeseries_interp[:, :, participant]
        )


def check_graph(G, weights):
    print(f"weights symmetric: {np.allclose(weights, weights.T)}")
    print(weights)

    # create graph for first participant & check properties
    G = graphs.Graph(weights)
    print("{} nodes, {} edges".format(G.N, G.Ne))

    print(f"connected: {G.is_connected()}")
    print(f"directed: {G.is_directed()}")  # why is it directed?

    print(G.check_weights())
    print(f"weights as initialized: {(G.W == weights).all()}")
