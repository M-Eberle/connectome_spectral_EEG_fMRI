from scipy import io as sio
from pygsp import graphs
import numpy as np
import h5py
import scipy.interpolate as interp

# ToDo: make participant order & region order tests/ sorting prettier!


def sort_EEG_data(SC_path, EEG_regions_path, fMRI_data_path):

    # make this prettier!!

    # compare sorting: SC matrices, fMRI, EEG
    SC_order = np.hstack(
        sio.loadmat(SC_path)["freesurfer_roi_IDs"][0].flatten()
    ).astype("uint16")

    EEG_order = sio.loadmat(EEG_regions_path)["regionsMap"][:, 0]

    fMRI_order = np.hstack(
        sio.loadmat(fMRI_data_path)["freesurfer_roi_IDs"][0].flatten()
    ).astype("uint16")

    N_regions = len(SC_order)
    region_sort_EEG_after_SC = np.empty((N_regions), dtype=int)
    for i in np.arange(N_regions):
        region_sort_EEG_after_SC[i] = np.argwhere(EEG_order == SC_order[i])

    
    for i in np.arange(len(SC_order)):
        print(SC_order[i])
        print(EEG_order[region_sort_EEG_after_SC][i])
        print(fMRI_order[i])
        print('\n')
    
    return region_sort_EEG_after_SC


def get_data_ind_SCs(SC_path, EEG_data_path, fMRI_data_path, EEG_regions_path, coords):
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
    N_regions = 68
    fMRI_timesteps = 661
    EEG_timesteps = 259184

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
    # arrays of signal timeseries
    # EEG: (68x259184)x15 for (RegionsxTime)xparticipants
    EEG_timeseries = np.empty((N_regions, EEG_timesteps, N))
    trans_EEG_timeseries = np.empty((N_regions, EEG_timesteps, N))
    # fMRI: before interpolation: (68x661)x15 for (RegionsxTime)xparticipants
    # after interpolation as in EEG: (68x259184)x15 for (RegionsxTime)xparticipants
    fMRI_timeseries = np.empty((N_regions, fMRI_timesteps, N))
    fMRI_timeseries_interp = np.empty((N_regions, EEG_timesteps, N))
    trans_fMRI_timeseries = np.empty((N_regions, EEG_timesteps, N))

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
        #EEG_timeseries_curr = EEG_timeseries_curr - np.mean(EEG_timeseries_curr, 0)
        EEG_timeseries[:, :, participant] = EEG_timeseries_curr[:, :].T

        # get participant's fMRI data
        fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
        fMRI_timeseries[:, :, participant] = fMRI_timeseries_curr.T

        # stretch fMRI data over time to EEG sequence length
        fMRI_interp_curr = np.empty((N_regions, EEG_timesteps))
        for region in np.arange(N_regions):
            fMRI_interp = interp.interp1d(
                np.arange(fMRI_timesteps), fMRI_timeseries[region, :, participant]
            )
            fMRI_interp_curr[region, :] = fMRI_interp(
                np.linspace(0, fMRI_timesteps - 1, EEG_timesteps)
            )
        fMRI_timeseries_interp[:, :, participant] = fMRI_interp_curr

    # !!
    # sort EEG data
    EEG_timeseries = EEG_timeseries[
        sort_EEG_data(SC_path, EEG_regions_path, fMRI_data_path), :, :
    ]

    for participant in np.arange(N):
        # spectral representation of signal
        # the Fourier transform is simply calling the .gft() method
        # e.g. G.gft(signal)
        trans_EEG_timeseries[:, :, participant] = Gs[participant].gft(
            EEG_timeseries[:, :, participant]
        )
        trans_fMRI_timeseries[:, :, participant] = Gs[participant].gft(
            fMRI_timeseries_interp[:, :, participant]
        )

        """
        # trans_fMRI_timeseries.append(G.gft(fMRI_timeseries[-1])) # for comparison: power stronger also high for low harmonics

        if (str(SC_participant_ID) == str(EEG_participant_ID)) and (
            str(SC_participant_ID) == str(fMRI_participant_ID)
        ):
            ID_count += 1
            if ID_count == N:
                print(
                    "all participant IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
                )
        """
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


def get_data_mean_SC(SC_path, EEG_data_path, fMRI_data_path, EEG_regions_path, coords):
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

    N_regions = 68
    fMRI_timesteps = 661
    EEG_timesteps = 259184

    # load SC data
    unflattened_SC_data = sio.loadmat(SC_path)["SC"]
    SC_data = np.ndarray.flatten(unflattened_SC_data)
    N = SC_data.size

    # load fMRI data
    unflattened_fMRI_data = sio.loadmat(fMRI_data_path)["fMRI"]
    fMRI_data = np.ndarray.flatten(unflattened_fMRI_data)

    # load EEG data
    EEG_data_file = h5py.File(EEG_data_path, "r")

    # array of SC matrices
    SC_weights = np.empty((68, 68, 15))
    # arrays of signal timeseries
    # EEG: (68x259184)x15 for (RegionsxTime)xparticipants
    EEG_timeseries = np.empty((N_regions, EEG_timesteps, N))
    trans_EEG_timeseries = np.empty((N_regions, EEG_timesteps, N))
    # fMRI: before interpolation: (68x661)x15 for (RegionsxTime)xparticipants
    # after interpolation as in EEG: (68x259184)x15 for (RegionsxTime)xparticipants
    fMRI_timeseries = np.empty((N_regions, fMRI_timesteps, N))
    fMRI_timeseries_interp = np.empty((N_regions, EEG_timesteps, N))
    trans_fMRI_timeseries = np.empty((N_regions, EEG_timesteps, N))

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
        EEG_timeseries[:, :, participant] = EEG_timeseries_curr[:, :].T

        # get participant's fMRI data
        fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
        fMRI_timeseries[:, :, participant] = fMRI_timeseries_curr.T

        # stretch fMRI data over time to EEG sequence length
        fMRI_interp_curr = np.empty((N_regions, EEG_timesteps))
        for region in np.arange(N_regions):
            fMRI_interp = interp.interp1d(
                np.arange(fMRI_timesteps), fMRI_timeseries[region, :, participant]
            )
            fMRI_interp_curr[region, :] = fMRI_interp(
                np.linspace(0, fMRI_timesteps - 1, EEG_timesteps)
            )
        fMRI_timeseries_interp[:, :, participant] = fMRI_interp_curr

    # !!
    # sort EEG data
    EEG_timeseries = EEG_timeseries[
        sort_EEG_data(SC_path, EEG_regions_path, fMRI_data_path), :, :
    ]

    for participant in np.arange(N):
        # spectral representation of signal
        # the Fourier transform is simply calling the .gft() method
        # e.g. G.gft(signal)
        trans_EEG_timeseries[:, :, participant] = G.gft(
            EEG_timeseries[:, :, participant]
        )
        trans_fMRI_timeseries[:, :, participant] = G.gft(
            fMRI_timeseries_interp[:, :, participant]
        )

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
