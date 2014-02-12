# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engmeann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
from ..externals.six import string_types

from .tree import dir_tree_find
from .tag import find_tag
from .constants import FIFF
from .pick import channel_type, pick_info
from ..utils import verbose, logger


def read_bad_channels(fid, node):
    """Read bad channels

    Parameters
    ----------
    fid : file
        The file descriptor.

    node : dict
        The node of the FIF tree that contains info on the bad channels.

    Returns
    -------
    bads : list
        A list of bad channel's names.
    """
    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = []
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag is not None and tag.data is not None:
                bads = tag.data.split(':')
    return bads


def _get_meg_system(info):
    """Educated guess for the helmet type based on channels"""
    system = '306m'
    for ch in info['chs']:
        if ch['kind'] == FIFF.FIFFV_MEG_CH:
            coil_type = ch['coil_type'] & 0xFFFF
            if coil_type == FIFF.FIFFV_COIL_NM_122:
                system = '122m'
                break
            elif coil_type // 1000 == 3:  # All Vectorview coils are 30xx
                system = '306m'
                break
            elif (coil_type == FIFF.FIFFV_COIL_MAGNES_MAG or
                  coil_type == FIFF.FIFFV_COIL_MAGNES_GRAD):
                nmag = np.sum([c['kind'] == FIFF.FIFFV_MEG_CH
                               for c in info['chs']])
                system = 'Magnes_3600wh' if nmag > 150 else 'Magnes_2500wh'
                break
            elif coil_type == FIFF.FIFFV_COIL_CTF_GRAD:
                system = 'CTF_275'
                break
            elif coil_type == FIFF.FIFFV_COIL_KIT_GRAD:
                system = 'KIT'
                break
            elif coil_type == FIFF.FIFFV_COIL_BABY_GRAD:
                system = 'BabySQUID'
                break
    return system


def _contains_ch_type(info, ch_type):
    """Check whether a certain channel type is in an info object

    Parameters
    ---------
    info : instance of mne.fiff.meas_info.Info
        The measurement information.
    ch_type : str
        the channel type to be checked for

    Returns
    -------
    has_ch_type : bool
        Whether the channel type is present or not.
    """
    if not isinstance(ch_type, string_types):
        raise ValueError('`ch_type` is of class {actual_class}. It must be '
                         '`str`'.format(actual_class=type(ch_type)))

    valid_channel_types = ('grad mag eeg stim eog emg ecg ref_meg resp '
                           'exci ias syst misc').split()

    if ch_type not in valid_channel_types:
        msg = ('The ch_type passed ({passed}) is not valid. '
               'it must be {valid}')
        raise ValueError(msg.format(passed=ch_type,
                                    valid=' or '.join(valid_channel_types)))
    return ch_type in [channel_type(info, ii) for ii in range(info['nchan'])]

@verbose
def equalize_channels(candidates, verbose=None):
    """Equalize channel picks for a collection of MNE-Python objects
    
    Parameters
    ----------
    candidates : list
        list Raw | Epochs | Evoked.
    verbose : None | bool
        whether to be verbose or not.

    Note. This function operates inplace.
    """
    from . import Raw
    from .. import Epochs
    from . import Evoked

    if not all([isinstance(c, (Raw, Epochs, Evoked)) for c in candidates]):
        valid = ['Raw', 'Epochs', 'Evoked']
        raise ValueError('candidates must be ' + ' or '.join(valid))
    
    chan_max_idx = np.argmax([c.info['nchan'] for c in candidates])
    chan_template = candidates[chan_max_idx].ch_names
    logger.info('Identiying common channels ...')
    channels = [set(c.ch_names) for c in candidates]
    common_channels = set(chan_template).intersection(*channels)
    dropped = list()
    for c in candidates:
        drop_them = list(set(c.ch_names) - common_channels)
        if drop_them:
            c.drop_channels(drop_them)
            dropped.extend(drop_them)
    if dropped:
        dropped = list(set(dropped))
        logger.info('Dropped the following channels:\n%s' % dropped)
    else:
        logger.info('all channels are corresponding, nothing to do.')

class ContainsMixin(object):
    """Mixin class for Raw, Evoked, Epochs
    """
    def __contains__(self, ch_type):
        """Check channel type membership"""
        if ch_type == 'meg':
            has_ch_type = (_contains_ch_type(self.info, 'mag') or
                           _contains_ch_type(self.info, 'grad'))
        else:
            has_ch_type = _contains_ch_type(self.info, ch_type)
        return has_ch_type

class DropChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs
    """

    def drop_channels(self, ch_names):
        """Drop some channels

        Parameters
        ----------
        ch_names : list
            The list of channels to remove.
        """
        # avoid circular imports
        from . import Raw
        from .. import Epochs
        from . import Evoked

        bad_idx = [self.ch_names.index(c) for c in ch_names]
        idx = np.setdiff1d(np.arange(len(self.ch_names)), bad_idx)
        if hasattr(self, 'picks'):
            self.picks = [self.picks[k] for k in idx]

        self.info = pick_info(self.info, idx, copy=False)
        
        my_get = lambda attr: getattr(self, attr, None)
        
        if my_get('_projector') is not None:
            self._projector = self._projector[idx][:, idx]

        if isinstance(self, Raw) and my_get('_preloaded'):
            self._data = self._data[idx, :]
        elif isinstance(self, Epochs) and my_get('preload'):
            self._data = self._data[:, idx, :]
        elif isinstance(self, Evoked):
            self.data = self.data[idx, :]


def rename_channels(info, alias):
    """Rename channels and optionally change the sensor type.

    Note: This only changes between the following sensor types: eeg, eog,
    emg, ecg, and misc. It also cannot change to eeg.

    Parameters
    ----------
    info : dict
        Measurement info.
    alias : dict
        a dictionary mapping the old channel to a new channel name {'EEG061' :
        'EEG161'}. If changing the sensor type, make the new name a tuple with
        the name (str) and the new channel type (str)
        {'EEG061',('EOG061','eog')}.
    """
    human2fiff = {'eeg': FIFF.FIFFV_EEG_CH,
                  'eog': FIFF.FIFFV_EOG_CH,
                  'emg': FIFF.FIFFV_EMG_CH,
                  'ecg': FIFF.FIFFV_ECG_CH,
                  'misc': FIFF.FIFFV_MISC_CH}
    ch_names = info['ch_names']
    bads = info['bads']
    chs = info['chs']
    info['ch_names'] = [ch['ch_name'] for ch in chs]  # reset for safety
    ch_names = info['ch_names']
    for ch_name in alias.keys():
        if ch_name not in ch_names:
            raise RuntimeError("This channel name %s doesn't exist in info."
                               % ch_name)
        else:
            c_ind = ch_names.index(ch_name)
            if type(alias[ch_name]) is str:  # just name change
                chs[c_ind]['ch_name'] = alias[ch_name]
                if ch_name in bads:  # check bads
                    bads[bads.index(ch_name)] = alias[ch_name]
            elif type(alias[ch_name]) is tuple:  # name and type change
                fiff_accept = human2fiff.values()
                fiff_accept.append(chs[c_ind]['kind'])
                if (len(fiff_accept) > len(set(fiff_accept))):
                    if alias[ch_name][1] in human2fiff:
                        chs[c_ind]['ch_name'] = alias[ch_name][0]
                        if ch_name in bads:  # check bads
                            bads[bads.index(ch_name)] = alias[ch_name][0]
                        chs[c_ind]['kind'] = human2fiff[alias[ch_name][1]]
                        if chs[c_ind]['kind'] is human2fiff['eeg']:
                            raise RuntimeError('This function cannot create '
                                               'eeg channels!')
                    else:
                        raise RuntimeError('This function cannot change to '
                                           'this channel type: %s.' %
                                           alias[ch_name][1])
                else:
                    raise RuntimeError('This function will not change from'
                                       ' this channel type. Please check'
                                       ' that this is the channel you mean.'
                                       )
            else:
                raise RuntimeError('Your alias is not configured properly. '
                                   'Please see the help: mne.rename_channels?')

    # Reset ch_names and Check that all the channel names are unique.
    info['chs'] = chs
    info['ch_names'] = [ch['ch_name'] for ch in chs]
    if (len(info['ch_names']) > len(set(info['ch_names']))):
        raise RuntimeError('You have created duplicate channel names. '
                           'Please check that your not renaming a channel to a'
                           ' name that already exists in ch_names')
