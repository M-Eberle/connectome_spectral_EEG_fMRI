# %% [markdown]

### last steps
# - load data
# - transform data onto graph
# - check that participant's indices in SC, fMRI, and EEG align
# - symmetry of SCs? --> use +transpose for now
### next steps
# - compare fMRI & EEG signal with individual SCs
#   - plot power over smoothness per participant
#   - nr. of harmonics needed to recreate fMRI/EEG signal --> cumulative power from Glomb et al. (2020) Fig. 2
#   - compare patterns between participants (correlation matrices?)
# - compare fMRI & EEG signal with average SC
#   - averaged correlation matrix ?
#   - ? plot power over smoothness per participant
#### other ToDos
# - fix time axis in all plots
# - save plots
# - plot signal on graph nodes

# %%
SC_path = "../data/empirical_structural_connectomes/SCs.mat"
fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
EEG_data_path = "../data/empirical_source_activity/source_activity.mat"

ex_participant = 3
ex_harmonic = 5
ex_region = 5

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
EEG_power = power_mean(trans_EEG_timeseries, ex_participant, "EEG")
fMRI_power = power_mean(trans_fMRI_timeseries, ex_participant, "fMRI")

# %%
plot_cum_power(EEG_power, "EEG")
plot_cum_power(fMRI_power, "fMRI")
