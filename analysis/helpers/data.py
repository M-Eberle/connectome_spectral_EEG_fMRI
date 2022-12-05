from scipy import io as sio
from pygsp import graphs
import numpy as np
import h5py
import scipy.interpolate as interp


def get_data_ind_SCs(SC_path, EEG_data_path, fMRI_data_path, coords):
    """
    SC matrices, EEG and fMRI timeseries from ‘Hybrid Brain Model data’ in data folder
    are extracted for all participants.
    The fMRI data is interpolated (stretched) to fit the length of the EEG data.
    The EEG and interpolated fMRI timeseries are transformed by the SC graph's
    Laplacian's eigenvectors.
    arguments:
        SC_path: path to SC matrices file
        EEG_data_path: path to EEG data file
        fMRI_data_path: path to fMRI data file
    returns:
        SC_weights: list with SC matrix for each participant
        EEG_timeseries: list with EEG activity for each participant
        trans_EEG_timeseries: list with EEG GFT weights for each participant
        fMRI_timeseries: list with fMRI activity for each participant
        fMRI_timeseries_interp: list with interpolated fMRI activity for each participant
        trans_fMRI_timeseries: list with fMRI GFT weights for each participant
    """
    # load SC data
    unflattened_SC_data = sio.loadmat(SC_path)["SC"]
    SC_data = np.ndarray.flatten(unflattened_SC_data)
    N = SC_data.size

    # load fMRI data
    unflattened_fMRI_data = sio.loadmat(fMRI_data_path)["fMRI"]
    fMRI_data = np.ndarray.flatten(unflattened_fMRI_data)

    # load EEG data
    EEG_data_file = h5py.File(EEG_data_path, "r")

    # list of SC matrices
    SC_weights = []
    Gs = []
    # lists of signal timeseries
    # EEG: 15x (68x259184) for participants x(RegionsxTime)
    EEG_timeseries = []
    trans_EEG_timeseries = []
    # fMRI: before interpolation: 15x (68x661) for participants x(RegionsxTime)
    # after interpolation as in EEG: 15x (68x259184) for participants x(RegionsxTime)
    fMRI_timeseries = []
    fMRI_timeseries_interp = []
    trans_fMRI_timeseries = []

    ID_count = 0
    for participant in np.arange(N):

        SC_weights_pre, SC_participant_ID = SC_data[participant]
        # extract weights from nested object
        # make weights symmetric to keep going for now
        SC_weights_pre[0, 0][0] = SC_weights_pre[0, 0][0] + SC_weights_pre[0, 0][0].T
        SC_weights.append(SC_weights_pre[0, 0][0])

        # compute participant's graph
        # save all Graphs?
        G = graphs.Graph(SC_weights[participant], lap_type="normalized", coords=coords)
        G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e
        Gs.append(G)

        # get participant's EEG data
        EEG_ID_store_curr = EEG_data_file["source_activity/sub_id"][participant][0]
        EEG_participant_ID = EEG_data_file[EEG_ID_store_curr][:]
        EEG_participant_ID = (
            "['" + "".join(chr(c[0]) for c in EEG_participant_ID) + "']"
        )

        EEG_data_store_curr = EEG_data_file["source_activity/ts"][participant][0]
        EEG_timeseries_curr = EEG_data_file[EEG_data_store_curr]
        # subtract mean over regions ?
        EEG_timeseries_curr = EEG_timeseries_curr - np.mean(EEG_timeseries_curr, 0)
        EEG_timeseries.append(EEG_timeseries_curr[:, :].T)

        # get participant's fMRI data
        fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
        fMRI_timeseries.append(fMRI_timeseries_curr.T)

        N_regions, fMRI_timesteps = fMRI_timeseries[-1].shape
        N_regions, EEG_timesteps = EEG_timeseries[-1].shape

        # stretch fMRI data over time to EEG sequence length
        fMRI_interp_curr = np.empty((N_regions, EEG_timesteps))
        for region in np.arange(N_regions):
            fMRI_interp = interp.interp1d(
                np.arange(fMRI_timesteps), fMRI_timeseries[-1][region, :]
            )
            fMRI_interp_curr[region, :] = fMRI_interp(
                np.linspace(0, fMRI_timesteps - 1, EEG_timesteps)
            )
        fMRI_timeseries_interp.append(fMRI_interp_curr)

        # spectral representation of signal
        # the Fourier transform is simply calling the .gft() method
        # e.g. G.gft(signal)
        trans_EEG_timeseries.append(G.gft(EEG_timeseries[-1]))
        trans_fMRI_timeseries.append(G.gft(fMRI_timeseries_interp[-1]))
        # trans_fMRI_timeseries.append(G.gft(fMRI_timeseries[-1])) # for comparison: power stronger also high for low harmonics

        if (str(SC_participant_ID) == str(EEG_participant_ID)) and (
            str(SC_participant_ID) == str(fMRI_participant_ID)
        ):
            ID_count += 1
            if ID_count == N:
                print(
                    "all participant IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
                )
    return (
        Gs,
        SC_weights,
        EEG_timeseries,
        trans_EEG_timeseries,
        fMRI_timeseries,
        fMRI_timeseries_interp,
        trans_fMRI_timeseries,
        N,
        N_regions,
        EEG_timesteps,
    )


def get_data_mean_SC(SC_path, EEG_data_path, fMRI_data_path, coords):
    """
    SC matrices, EEG and fMRI timeseries from ‘Hybrid Brain Model data’ in data folder
    are extracted for all participants.
    The fMRI data is interpolated (stretched) to fit the length of the EEG data.
    The EEG and interpolated fMRI timeseries are transformed by the SC graph's
    Laplacian's eigenvectors.
    arguments:
        SC_path: path to SC matrices file
        EEG_data_path: path to EEG data file
        fMRI_data_path: path to fMRI data file
    returns:
        mean_SC_weights: mean SC matrix
        G: graph generated from SC matrix
        EEG_timeseries: list with EEG activity for each participant
        trans_EEG_timeseries: list with EEG GFT weights for each participant
        fMRI_timeseries: list with fMRI activity for each participant
        fMRI_timeseries_interp: list with interpolated fMRI activity for each participant
        trans_fMRI_timeseries: list with fMRI GFT weights for each participant
    """
    # load SC data
    unflattened_SC_data = sio.loadmat(SC_path)["SC"]
    SC_data = np.ndarray.flatten(unflattened_SC_data)
    N = SC_data.size

    # load fMRI data
    unflattened_fMRI_data = sio.loadmat(fMRI_data_path)["fMRI"]
    fMRI_data = np.ndarray.flatten(unflattened_fMRI_data)

    # load EEG data
    EEG_data_file = h5py.File(EEG_data_path, "r")

    # list of SC matrices
    SC_weights = np.empty((68, 68, 15))
    # lists of signal timeseries
    # EEG: 15x (68x259184) for participants x(RegionsxTime)
    EEG_timeseries = []
    trans_EEG_timeseries = []
    # fMRI: before interpolation: 15x (68x661) for participants x(RegionsxTime)
    # after interpolation as in EEG: 15x (68x259184) for participants x(RegionsxTime)
    fMRI_timeseries = []
    fMRI_timeseries_interp = []
    trans_fMRI_timeseries = []

    ID_count = 0
    for participant in np.arange(N):

        SC_weights_pre, SC_participant_ID = SC_data[participant]
        # extract weights from nested object
        # make weights symmetric to keep going for now
        SC_weights_pre[0, 0][0] = SC_weights_pre[0, 0][0] + SC_weights_pre[0, 0][0].T
        SC_weights[:, :, participant] = SC_weights_pre[0, 0][0]
    mean_SC_weights = np.mean(SC_weights, 2)

    # compute one graph for all participants
    G = graphs.Graph(mean_SC_weights, lap_type="normalized", coords=coords)
    G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e

    for participant in np.arange(N):

        # get participant's EEG data
        EEG_ID_store_curr = EEG_data_file["source_activity/sub_id"][participant][0]
        EEG_participant_ID = EEG_data_file[EEG_ID_store_curr][:]
        EEG_participant_ID = (
            "['" + "".join(chr(c[0]) for c in EEG_participant_ID) + "']"
        )

        EEG_data_store_curr = EEG_data_file["source_activity/ts"][participant][0]
        EEG_timeseries_curr = EEG_data_file[EEG_data_store_curr]
        # subtract mean over regions ?
        EEG_timeseries_curr = EEG_timeseries_curr - np.mean(EEG_timeseries_curr, 0)
        EEG_timeseries.append(EEG_timeseries_curr[:, :].T)

        # get participant's fMRI data
        fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
        fMRI_timeseries.append(fMRI_timeseries_curr.T)

        N_regions, fMRI_timesteps = fMRI_timeseries[-1].shape
        N_regions, EEG_timesteps = EEG_timeseries[-1].shape

        # stretch fMRI data over time to EEG sequence length
        fMRI_interp_curr = np.empty((N_regions, EEG_timesteps))
        for region in np.arange(N_regions):
            fMRI_interp = interp.interp1d(
                np.arange(fMRI_timesteps), fMRI_timeseries[-1][region, :]
            )
            fMRI_interp_curr[region, :] = fMRI_interp(
                np.linspace(0, fMRI_timesteps - 1, EEG_timesteps)
            )
        fMRI_timeseries_interp.append(fMRI_interp_curr)

        # spectral representation of signal
        # the Fourier transform is simply calling the .gft() method
        # e.g. G.gft(signal)
        trans_EEG_timeseries.append(G.gft(EEG_timeseries[-1]))
        trans_fMRI_timeseries.append(G.gft(fMRI_timeseries_interp[-1]))

        if (str(SC_participant_ID) == str(EEG_participant_ID)) and (
            str(SC_participant_ID) == str(fMRI_participant_ID)
        ):
            ID_count += 1
            if ID_count == N:
                print(
                    "all participant IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
                )
    return (
        G,
        mean_SC_weights,
        EEG_timeseries,
        trans_EEG_timeseries,
        fMRI_timeseries,
        fMRI_timeseries_interp,
        trans_fMRI_timeseries,
        N,
        N_regions,
        EEG_timesteps,
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
