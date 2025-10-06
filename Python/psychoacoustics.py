"""
Psychoacoustic Scale Conversion Functions

This module provides functions to convert between frequency (Hz) and various psychoacoustic scales:
- Mel scale
- ERB (Equivalent Rectangular Bandwidth) scale
- Bark scale

References and usage notes are included in the docstrings of each function.
"""
import numpy as np

def freq2mel(freq):
    """
    Frequency to Mel conversion
    
    The mel scale is a linearization of the pitch space perception. The idea is that musical intervals are not exactly logarithmically spaced in the frequency axis. Accordingly, a span of 4 perfect octaves (2^4) is perceived as wider than what 4 musical octaves should be, for example. Psychoacoustical analysis was performed to obtain statistical proof of said pitch space compression.
    
    This mainly applies for perception of pitch space and does not take into account the implications of more complex tones.
    
    (Notice that this concept has been criticized even by one of its main authors and needs further revision. It is possible that other scales might be more relevant.)
    
    Parameters:
        freq (float or array-like): Frequency in Hz
    Returns:
        float or ndarray: Value(s) in Mel scale
    """
    freq = np.asarray(freq)
    return 1127 * np.log1p(freq / 700)

def mel2freq(mel):
    """
    Mel to frequency conversion
    
    See freq2mel for details on the Mel scale.
    
    Parameters:
        mel (float or array-like): Value(s) in Mel scale
    Returns:
        float or ndarray: Frequency in Hz
    """
    mel = np.asarray(mel)
    return 700 * (np.exp(mel / 1127) - 1)

def freq2erb(freq):
    """
    Frequency to Equivalent Rectangular Bandwidth (ERB) conversion
    
    The ERB scale is a model of listening critical bands where a full ERB shares particular perceptual characteristics based on masking curves, equal loudness perception, and even complex tone integration. Notice however that any of these can be affected by other objective signal properties like loudness and timbre.
    
    Reference:
    [1] Moore and Glasberg "A Revision of Zwicker's Loudness Model," ACTA Acustica, vol. 82, pp. 335-345, 1996
    
    Parameters:
        freq (float or array-like): Frequency in Hz
    Returns:
        float or ndarray: Value(s) in ERB scale
    """
    freq = np.asarray(freq)
    return 21.4 * np.log10(4.37 * freq / 1000 + 1.0)

def erb2freq(erb):
    """
    Equivalent Rectangular Bandwidth (ERB) to frequency conversion
    
    See freq2erb for details on the ERB scale.
    
    Parameters:
        erb (float or array-like): Value(s) in ERB scale
    Returns:
        float or ndarray: Frequency in Hz
    """
    erb = np.asarray(erb)
    return 1000 * (10 ** (erb / 21.4) - 1.0) / 4.37

def freq2bark(freq):
    """
    Frequency to Bark scale conversion
    
    The bark scale is a psychoacoustically defined scale of perception of frequency space in which the perception of such space is linearized. The idea behind it is that human perception is based on critical bands of hearing that share specific behaviors. The bandwidth of such frequencies is not constant across frequency, so the purpose of the bark scale is to quantify such bandwidth across frequencies.
    
    Reference: H. Traunmüller (1990) "Analytical expressions for the tonotopic sensory scale" J. Acoust. Soc. Am. 88: 97-100.
    
    Parameters:
        freq (float or array-like): Frequency in Hz
    Returns:
        float or ndarray: Value(s) in Bark scale
    """
    freq = np.asarray(freq)
    return (26.81 / (1 + 1960 / freq)) - 0.53

def bark2freq(bark):
    """
    Bark scale to frequency conversion
    
    See freq2bark for details on the Bark scale.
    
    Parameters:
        bark (float or array-like): Value(s) in Bark scale
    Returns:
        float or ndarray: Frequency in Hz
    """
    bark = np.asarray(bark)
    return 1960 / (26.81 / (bark + 0.53) - 1)
