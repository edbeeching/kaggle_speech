# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:08:26 2017

@author: Edward
"""

from __future__ import absolute_import
import numpy as np
import keras
from keras import backend as K
from keras.engine import Layer
from keras.utils.conv_utils import conv_output_length


class Spectrogram(Layer):
    """
    ### `Spectrogram`
    ```python
    kapre.time_frequency.Spectrogram(n_dft=512, n_hop=None, padding='same',
                                     power_spectrogram=2.0, return_decibel_spectrogram=False,
                                     trainable_kernel=False, image_data_format='default',
                                     **kwargs)
    ```
    Spectrogram layer that outputs spectrogram(s) in 2D image format.
    #### Parameters
     * n_dft: int > 0 [scalar]
       - The number of DFT points, presumably power of 2.
       - Default: ``512``
     * n_hop: int > 0 [scalar]
       - Hop length between frames in sample,  probably <= ``n_dft``.
       - Default: ``None`` (``n_dft / 2`` is used)
     * padding: str, ``'same'`` or ``'valid'``.
       - Padding strategies at the ends of signal.
       - Default: ``'same'``
     * power_spectrogram: float [scalar],
       -  ``2.0`` to get power-spectrogram, ``1.0`` to get amplitude-spectrogram.
       -  Usually ``1.0`` or ``2.0``.
       -  Default: ``2.0``
     * return_decibel_spectrogram: bool,
       -  Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if ``True``.
       -  Recommended to use ``True``, although it's not by default.
       -  Default: ``False``
     * trainable_kernel: bool
       -  Whether the kernels are trainable or not.
       -  If ``True``, Kernels are initialised with DFT kernels and then trained.
       -  Default: ``False``
     * image_data_format: string, ``'channels_first'`` or ``'channels_last'``.
       -  The returned spectrogram follows this image_data_format strategy.
       -  If ``'default'``, it follows the current Keras session's setting.
       -  Setting is in ``./keras/keras.json``.
       -  Default: ``'default'``
    #### Notes
     * The input should be a 2D array, ``(audio_channel, audio_length)``.
     * E.g., ``(1, 44100)`` for mono signal, ``(2, 44100)`` for stereo signal.
     * It supports multichannel signal input, so ``audio_channel`` can be any positive integer.
    #### Returns
    A Keras layer
     * abs(Spectrogram) in a shape of 2D data, i.e.,
     * `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
     * `(None, n_freq, n_time, n_channel)` if `'channels_last'`,
    """

    def __init__(self, n_dft=512, n_hop=None, padding='same',
                 power_spectrogram=2.0, return_decibel_spectrogram=False,
                 trainable_kernel=False, image_data_format='default', **kwargs):
        assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
            ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)
        assert isinstance(trainable_kernel, bool)
        assert isinstance(return_decibel_spectrogram, bool)
        assert padding in ('same', 'valid')
        if n_hop is None:
            n_hop = n_dft // 2

        assert image_data_format in ('default', 'channels_first', 'channels_last')
        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        self.n_dft = n_dft
        assert n_dft % 2 == 0
        self.n_filter = int(n_dft // 2) + 1
        self.trainable_kernel = trainable_kernel
        self.n_hop = n_hop
        self.padding = 'same'
        self.power_spectrogram = float(power_spectrogram)
        self.return_decibel_spectrogram = return_decibel_spectrogram
        super(Spectrogram, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_ch = input_shape[1]
        self.len_src = input_shape[2]
        self.is_mono = (self.n_ch == 1)
        if self.image_data_format == 'channels_first':
            self.ch_axis_idx = 1
        else:
            self.ch_axis_idx = 3
        if self.len_src is not None:
            assert self.len_src >= self.n_dft, 'Hey! The input is too short!'

        self.n_frame = conv_output_length(self.len_src,
                                          self.n_dft,
                                          self.padding,
                                          self.n_hop)

        dft_real_kernels, dft_imag_kernels = _get_stft_kernels(self.n_dft)
        self.dft_real_kernels = K.variable(dft_real_kernels, dtype=K.floatx(), name="real_kernels")
        self.dft_imag_kernels = K.variable(dft_imag_kernels, dtype=K.floatx(), name="imag_kernels")
        # kernels shapes: (filter_length, 1, input_dim, nb_filter)?
        if self.trainable_kernel:
            self.trainable_weights.append(self.dft_real_kernels)
            self.trainable_weights.append(self.dft_imag_kernels)
        else:
            self.non_trainable_weights.append(self.dft_real_kernels)
            self.non_trainable_weights.append(self.dft_imag_kernels)

        super(Spectrogram, self).build(input_shape)
        # self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_filter, self.n_frame
        else:
            return input_shape[0], self.n_filter, self.n_frame, self.n_ch

    def call(self, x):
        output = self._spectrogram_mono(x[:, 0:1, :])
        if self.is_mono is False:
            for ch_idx in range(1, self.n_ch):
                output = K.concatenate((output,
                                        self._spectrogram_mono(x[:, ch_idx:ch_idx + 1, :])),
                                       axis=self.ch_axis_idx)
        if self.power_spectrogram != 2.0:
            output = K.pow(K.sqrt(output), self.power_spectrogram)
        if self.return_decibel_spectrogram:
            output = _amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'n_dft': self.n_dft,
                  'n_hop': self.n_hop,
                  'padding': self.padding,
                  'power_spectrogram': self.power_spectrogram,
                  'return_decibel_spectrogram': self.return_decibel_spectrogram,
                  'trainable_kernel': self.trainable_kernel,
                  'image_data_format': self.image_data_format}
        base_config = super(Spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _spectrogram_mono(self, x):
        '''x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram'''
        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.expand_dims(x, 3)  # add a dummy dimension (channel axis)
        subsample = (self.n_hop, 1)
        output_real = K.conv2d(x, self.dft_real_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output_imag = K.conv2d(x, self.dft_imag_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output = output_real ** 2 + output_imag ** 2
        # now shape is (batch_sample, n_frame, 1, freq)
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 3, 1, 2])
        else:
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        return output


class Melspectrogram(Spectrogram):
    '''
    ### `Melspectrogram`
    ```python
    kapre.time_frequency.Melspectrogram(sr=22050, n_mels=128, fmin=0.0, fmax=None,
                                        power_melgram=1.0, return_decibel_melgram=False,
                                        trainable_fb=False, **kwargs)
    ```
    Mel-spectrogram layer that outputs mel-spectrogram(s) in 2D image format.
    Its base class is ``Spectrogram``.
    Mel-spectrogram is an efficient representation using the property of human
    auditory system -- by compressing frequency axis into mel-scale axis.
    #### Parameters
     * sr: integer > 0 [scalar]
       - sampling rate of the input audio signal.
       - Default: ``22050``
     * n_mels: int > 0 [scalar]
       - The number of mel bands.
       - Default: ``128``
     * fmin: float > 0 [scalar]
       - Minimum frequency to include in Mel-spectrogram.
       - Default: ``0.0``
     * fmax: float > ``fmin`` [scalar]
       - Maximum frequency to include in Mel-spectrogram.
       - If `None`, it is inferred as ``sr / 2``.
       - Default: `None`
     * power_melgram: float [scalar]
       - Power of ``2.0`` if power-spectrogram,
       - ``1.0`` if amplitude spectrogram.
       - Default: ``1.0``
     * return_decibel_melgram: bool
       - Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if ``True``.
       - Recommended to use ``True``, although it's not by default.
       - Default: ``False``
     * trainable_fb: bool
       - Whether the spectrogram -> mel-spectrogram filterbanks are trainable.
       - If ``True``, the frequency-to-mel matrix is initialised with mel frequencies but trainable.
       - If ``False``, it is initialised and then frozen.
       - Default: `False`
     * **kwargs:
       - The keyword arguments of ``Spectrogram`` such as ``n_dft``, ``n_hop``,
       - ``padding``, ``trainable_kernel``, ``image_data_format``.
    #### Notes
     * The input should be a 2D array, ``(audio_channel, audio_length)``.
    E.g., ``(1, 44100)`` for mono signal, ``(2, 44100)`` for stereo signal.
     * It supports multichannel signal input, so ``audio_channel`` can be any positive integer.
    #### Returns
    A Keras layer
     * abs(mel-spectrogram) in a shape of 2D data, i.e.,
     * `(None, n_channel, n_mels, n_time)` if `'channels_first'`,
     * `(None, n_mels, n_time, n_channel)` if `'channels_last'`,
    '''

    def __init__(self,
                 sr=22050, n_mels=128, fmin=0.0, fmax=None,
                 power_melgram=1.0, return_decibel_melgram=False,
                 trainable_fb=False, **kwargs):

        super(Melspectrogram, self).__init__(**kwargs)
        assert sr > 0
        assert fmin >= 0.0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin
        assert isinstance(return_decibel_melgram, bool)
        if 'power_spectrogram' in kwargs:
            assert kwargs['power_spectrogram'] == 2.0, \
                'In Melspectrogram, power_spectrogram should be set as 2.0.'

        self.sr = int(sr)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.return_decibel_melgram = return_decibel_melgram
        self.trainable_fb = trainable_fb
        self.power_melgram = power_melgram

    def build(self, input_shape):
        super(Melspectrogram, self).build(input_shape)
        self.built = False
        # compute freq2mel matrix --> 
        mel_basis = K.mel(self.sr, self.n_dft, self.n_mels, self.fmin, self.fmax)  # (128, 1025) (mel_bin, n_freq)
        mel_basis = np.transpose(mel_basis)

        self.freq2mel = K.variable(mel_basis, dtype=K.floatx())
        if self.trainable_fb:
            self.trainable_weights.append(self.freq2mel)
        else:
            self.non_trainable_weights.append(self.freq2mel)
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_mels, self.n_frame
        else:
            return input_shape[0], self.n_mels, self.n_frame, self.n_ch

    def call(self, x):
        power_spectrogram = super(Melspectrogram, self).call(x)
        # now,  th: (batch_sample, n_ch, n_freq, n_time)
        #       tf: (batch_sample, n_freq, n_time, n_ch)
        if self.image_data_format == 'channels_first':
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 1, 3, 2])
        else:
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 3, 2, 1])
        # now, whatever image_data_format, (batch_sample, n_ch, n_time, n_freq)
        output = K.dot(power_spectrogram, self.freq2mel)
        if self.image_data_format == 'channels_first':
            output = K.permute_dimensions(output, [0, 1, 3, 2])
        else:
            output = K.permute_dimensions(output, [0, 3, 2, 1])
        if self.power_melgram != 2.0:
            output = K.pow(K.sqrt(output), self.power_melgram)
        if self.return_decibel_melgram:
            output = K.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'sr': self.sr,
                  'n_mels': self.n_mels,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'trainable_fb': self.trainable_fb,
                  'power_melgram': self.power_melgram,
                  'return_decibel_melgram': self.return_decibel_melgram}
        base_config = super(Melspectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def _get_stft_kernels(n_dft):
    '''[np] Return dft kernels for real/imagnary parts assuming
        the input . is real.
    An asymmetric hann window is used (scipy.signal.hann).
    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        Number of dft components.
    Returns
    -------
        |  dft_real_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
        |  dft_imag_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    * nb_filter = n_dft/2 + 1
    * n_win = n_dft
    '''
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)

    nb_filter = int(n_dft / 2 + 1)
    dtype = K.floatx()

    # prepare DFT filters
    timesteps = range(n_dft)
    w_ks = [(2 * np.pi * k) / float(n_dft) for k in range(n_dft)]
    dft_real_kernels = np.array([[np.cos(w_k * n) for n in timesteps] for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k * n) for n in timesteps] for w_k in w_ks])

    # windowing DFT filters
    dft_window = _hann(n_dft, sym=False)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    dft_real_kernels = dft_real_kernels[:nb_filter].transpose()
    dft_imag_kernels = dft_imag_kernels[:nb_filter].transpose()
    dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    return dft_real_kernels.astype(K.floatx()), dft_imag_kernels.astype(K.floatx())


def _hann(M, sym=True):
    '''[np] 
    Return a Hann window.
    copied and pasted from scipy.signal.hann,
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L615
    ----------
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).
    
    '''
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
    if not sym and not odd:
        w = w[:-1]
    return w.astype(K.floatx())


def _amplitude_to_decibel(x, amin=1e-10, dynamic_range=80.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).
    x: Keras tensor or variable. 
    amin: minimum amplitude. amplitude smaller than `amin` is set to this.
    dynamic_range: dynamic_range in decibel
    """
    log_spec = 10 * K.log(K.maximum(x, amin)) / np.log(10).astype(K.floatx())
    log_spec = log_spec - K.max(log_spec)  # [-?, 0]
    log_spec = K.maximum(log_spec, -1 * dynamic_range)  # [-80, 0]
    return log_spec


class Normalization2D(Layer):
    '''
    
    ### `Normalization2D`
    
    `kapre.utils.Normalization2D`
    
    A layer that normalises input data in ``axis`` axis.
    #### Parameters
    
    * input_shape: tuple of ints
        - E.g., ``(None, n_ch, n_row, n_col)`` if theano.
    * str_axis: str
        - used ONLY IF ``int_axis`` is ``None``.
        - ``'batch'``, ``'data_sample'``, ``'channel'``, ``'freq'``, ``'time')``
        - Even though it is optional, actually it is recommended to use
        - ``str_axis`` over ``int_axis`` because it provides more meaningful
        - and image data format-robust interface.
    * int_axis: int
        - axis index that along which mean/std is computed.
        - `0` for per data sample, `-1` for per batch.
        - `1`, `2`, `3` for channel, row, col (if channels_first)
        - if `int_axis is None`, ``str_axis`` SHOULD BE set.
    #### Example
    A frequency-axis normalization after a spectrogram::
        ```python
        model.add(Spectrogram())
        model.add(Normalization2D(stf_axis='freq'))
        ```
    '''

    def __init__(self, str_axis=None, int_axis=None, image_data_format='default',
                 eps=1e-10, **kwargs):
        assert not (int_axis is None and str_axis is None), \
            'In Normalization2D, int_axis or str_axis should be specified.'

        assert image_data_format in ('channels_first', 'channels_last', 'default'), \
            'Incorrect image_data_format: {}'.format(image_data_format)

        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        self.str_axis = str_axis
        if self.str_axis is None: # use int_axis
            self.int_axis = int_axis
        else: # use str_axis
            # warning
            if int_axis is not None:
                print('int_axis={} passed but is ignored, str_axis is used instead.'.format(int_axis))
            # do the work
            assert str_axis in ('batch', 'data_sample', 'channel', 'freq', 'time'), \
                'Incorrect str_axis: {}'.format(str_axis)
            if str_axis == 'batch':
                int_axis = -1
            else:
                if self.image_data_format == 'channels_first':
                    int_axis = ['data_sample', 'channel', 'freq', 'time'].index(str_axis)
                else:
                    int_axis = ['data_sample', 'freq', 'time', 'channel'].index(str_axis)

        assert int_axis in (-1, 0, 1, 2, 3), 'invalid int_axis: ' + str(int_axis)
        self.axis = int_axis
        self.eps = eps
        super(Normalization2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.axis == -1:
            mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
            std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = K.mean(x, axis=all_dims, keepdims=True)
            std = K.std(x, axis=all_dims, keepdims=True)
        return (x - mean) / (std + self.eps)

    def get_config(self):
        config = {'int_axis': self.axis,
                  'str_axis': self.str_axis,
                  'image_data_format': self.image_data_format}
        base_config = super(Normalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))