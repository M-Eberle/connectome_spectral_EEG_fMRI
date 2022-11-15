# %%
from scipy import io as sio
from pygsp import graphs, filters, plotting, utils
import numpy as np
import matplotlib.pyplot as plt
import h5py


# %% [markdown]

### last steps
# - load data
# - transform data onto graph
# - check that participant's indices in SC, fMRI, and EEG align
# - symmetry of SCs? --> use +transpose for now
### next steps
# - compare fMRI & EEG signal with individual SCs
#   - plot power over smoothness per participant
#   - nr. of harmonics needed to recreate fMRI/EEG signal
#   - compare patterns between participants
# - compare fMRI & EEG signal with average SC ?

# %%
SC_path = "../data/empirical_structural_connectomes/SCs.mat"
fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
EEG_data_path = "../data/empirical_source_activity/source_activity.mat"

# load SC data
unflattened_SC_data = sio.loadmat(SC_path)["SC"]
SC_data = np.ndarray.flatten(unflattened_SC_data)
N = SC_data.size

# load fMRI data
unflattened_fMRI_data = sio.loadmat(fMRI_data_path)["fMRI"]
fMRI_data = np.ndarray.flatten(unflattened_fMRI_data)

# load EEG data
EEG_data_file = h5py.File(EEG_data_path, "r")


# %%
ID_count = 0
# list of SC matrices
SC_weights = []
# lists of signal timeseries
# fMRI: 15x (68x661) for participants x(RegionsxTime)
fMRI_timeseries = []
trans_fMRI_timeseries = []
# EEG: 15x (68x259184) for participants x(RegionsxTime)
EEG_timeseries = []
trans_EEG_timeseries = []

N = 1
for participant in np.arange(N):

    SC_weights_pre, SC_participant_ID = SC_data[participant]
    # extract weights from nested object
    # make weights symmetric to keep going for now
    # ??
    SC_weights_pre[0, 0][0] = SC_weights_pre[0, 0][0] + SC_weights_pre[0, 0][0].T
    SC_weights.append(SC_weights_pre[0, 0][0])

    # compute participant's graph
    # save all Graphs?
    G = graphs.Graph(SC_weights[participant], lap_type="normalized")
    G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e

    # get participant's fMRI data
    fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
    fMRI_timeseries.append(fMRI_timeseries_curr.T)

    # get participant's EEG data
    EEG_ID_store_curr = EEG_data_file["source_activity/sub_id"][participant][0]
    EEG_participant_ID = EEG_data_file[EEG_ID_store_curr][:]
    EEG_participant_ID = "['" + "".join(chr(c[0]) for c in EEG_participant_ID) + "']"

    EEG_data_store_curr = EEG_data_file["source_activity/ts"][participant][0]
    EEG_timeseries_curr = EEG_data_file[EEG_data_store_curr]
    EEG_timeseries.append(EEG_timeseries_curr[:, :].T)

    # the Fourier transform is simply calling the .gft() method
    # e.g. G.gft(signal)
    trans_fMRI_timeseries.append(G.gft(fMRI_timeseries[-1]))
    trans_EEG_timeseries.append(G.gft(EEG_timeseries[-1]))

    if (str(SC_participant_ID) == str(EEG_participant_ID)) and (
        str(SC_participant_ID) == str(fMRI_participant_ID)
    ):
        ID_count += 1
        if ID_count == N:
            print(
                "all participant IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
            )

# %%
N, M = trans_fMRI_timeseries[-1].shape
for i in np.linspace(0, N - 1, 5).astype(int):
    plt.plot(trans_fMRI_timeseries[-1][i, :], label=f"harmonic {i+1}")
plt.xlabel("time")
plt.ylabel("signal")
plt.title("examplary fMRI activities for one participant")
plt.legend()
plt.show()

# %%
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


print(check_symmetric(SC_weights[0]))
print(SC_weights[0])

# create graph for first participant & check properties
G1 = graphs.Graph(SC_weights[0])
print("{} nodes, {} edges".format(G1.N, G1.Ne))

print(f"connected: {G1.is_connected()}")
print(f"directed: {G1.is_directed()}")  # why is it directed?

print(G1.check_weights())
print(f"weights as initialized: {(G1.W == SC_weights[0]).all()}")
