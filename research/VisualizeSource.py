import os
import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne import read_evokeds
import matplotlib
matplotlib.use('MacOSX')

# get the paths for the evoked data and the time courses
data_path = sample.data_path()
sample_dir = os.path.join(data_path, 'MEG', 'sample')
subjects_dir = os.path.join(data_path, 'subjects')

fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
fname_stc = os.path.join(sample_dir, 'sample_audvis-meg')

# The source estimate object
stc = mne.read_source_estimate(fname_stc, subject='sample')
print(stc)

initial_time = 0.1
brain = stc.plot(subjects_dir=subjects_dir, initial_time=initial_time,
                 clim=dict(kind='value', pos_lims=[3, 6, 9]),
                 time_viewer=True)

mpl_fig = stc.plot(subjects_dir=subjects_dir, initial_time=initial_time,
                   backend='matplotlib', verbose='error')


# Volume Source Estimates
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
evoked.pick_types(meg=True, eeg=False)

fname_inv = data_path + '/MEG/sample/sample_audvis-meg-vol-7-meg-inv.fif'
inv = read_inverse_operator(fname_inv)
src = inv['src']

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "sLORETA"  # use dSPM method (could also be MNE or sLORETA)
stc = apply_inverse(evoked, inv, lambda2, method)
stc.crop(0.0, 0.2)

stc.plot(src, subject='sample', subjects_dir=subjects_dir)
stc.plot(src, subject='sample', subjects_dir=subjects_dir, mode='glass_brain')


# # Vector Source Estimates
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'

inv = read_inverse_operator(fname_inv)
stc = apply_inverse(evoked, inv, lambda2, 'dSPM', pick_ori='vector')
stc.plot(subject='sample', subjects_dir=subjects_dir,
         initial_time=0.49948803289596966)


# Dipole fits
fname_cov = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_bem = os.path.join(subjects_dir, 'sample', 'bem',
                         'sample-5120-bem-sol.fif')
fname_trans = os.path.join(data_path, 'MEG', 'sample',
                           'sample_audvis_raw-trans.fif')

evoked.crop(0.1, 0.1)
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]
dip.plot_locations(fname_trans, 'sample', subjects_dir)


