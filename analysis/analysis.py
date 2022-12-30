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
# - data in numpy arrays instead of lists
# - general refactoring
# - scale timeseries before applying similarity measures
# - compare timeseries EEG & fMRI (e.g. lower vs higehr half of harmonics)
# - Fischer Transform instead of mean for correlations
# next steps
# - save similarity measures as Data class attributes (save time generating JET)
# - decide on suitable gamma for JET
#### other ToDos
# - fix time axis ticks in all plots / both axes ticks in heatmaps
# - save plots
# - compare scipy.interpolate.interp1d & scipy.signal.resample


# %%
from helpers import *

# currently, some plots/calculations only work for individual SC matrices
SC_mode = "ind"  # 'mean' or 'ind'

# looping over all participants for everything takes too long
data_ind = Data(SC_mode=SC_mode, loop_participants=False)

# %%
# overview plots
data_ind.plot_signal()
data_ind.plot_signal_single_domain()
data_ind.plot_domain()
data_ind.plot_power_stem_cum()
data_ind.plot_power_corr()

# sanity checks for EEG
# sanity check 1
data_ind.plot_power_mean_corr()
data_ind.plot_power_mean_stem_cum()
# sanity check 2
data_ind.plot_EEG_freq_band()
# sanity check 3
data_ind.get_alpha_corrs()
data_ind.plot_alpha_corrs()

# hypotheses
data_ind.get_vertex_vs_graph()
data_ind.plot_vertex_vs_graph()
data_ind.get_lower_vs_upper_harmonics()
data_ind.get_TVG()
data_ind.get_GE()
data_ind.get_JET()
