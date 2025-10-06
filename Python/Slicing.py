import numpy as np

def slicer(x, block_size, overlap_size):
    """
    Slice a 1D array into overlapping blocks.

    Parameters:
        x (np.ndarray): Input 1D array.
        block_size (int): Size of each block.
        overlap_size (int): Number of samples that overlap between blocks.

    Returns:
        np.ndarray: 2D array of shape (n_blocks, block_size) containing the blocks.
    """
    n_samps = x.size
    hop_size = block_size - overlap_size  # Step size between blocks
    # Compute indices for each block
    idx = np.add.outer(np.arange(0, n_samps - block_size + 1, hop_size), np.arange(block_size))
    return x[idx]

def splicer(x, overlap_size):
    """
    Reconstruct a 1D array from overlapping blocks (inverse of slicer).

    Parameters:
        x (np.ndarray): 2D array of shape (n_blocks, block_size) containing the blocks.
        overlap_size (int): Number of samples that overlap between blocks.

    Returns:
        np.ndarray: Reconstructed 1D array.
    """
    n_blocks, block_size = x.shape
    hop_size = block_size - overlap_size
    pos = 0

    n_samples = hop_size * n_blocks + overlap_size  # Total length of reconstructed signal
    y = np.zeros(n_samples, dtype='complex128')

    for i in np.arange(n_blocks):
        ii = np.arange(block_size) + pos  # Indices for current block
        y[ii] += x[i, :]
        pos += hop_size
    return y

def spectral_slicer(X, block_size, overlap_size):
    """
    Slice a 2D array (e.g., time-frequency representation) into overlapping blocks along the first axis.

    Parameters:
        X (np.ndarray): 2D array of shape (n_samples, n_bands).
        block_size (int): Size of each block.
        overlap_size (int): Number of samples that overlap between blocks.

    Returns:
        np.ndarray: 3D array of shape (n_blocks, n_bands, block_size) containing the blocks.
    """
    n_samps, n_bands = X.shape
    hop_size = block_size - overlap_size
    n_blocks = int(np.ceil((n_samps - overlap_size) / hop_size))
    X_block = np.zeros([n_blocks, n_bands, block_size], dtype='complex128')
    for k in np.arange(n_bands):
        # Slice each band independently
        X_block[:, k, :] = slicer(X[:, k], block_size, overlap_size)
    return X_block

def spectral_splicer(X_block, overlap_size):
    """
    Reconstruct a 2D array from overlapping blocks (inverse of spectral_slicer).

    Parameters:
        X_block (np.ndarray): 3D array of shape (n_blocks, n_bands, block_size) containing the blocks.
        overlap_size (int): Number of samples that overlap between blocks.

    Returns:
        np.ndarray: 2D array of shape (n_samples, n_bands) reconstructed from the blocks.
    """
    n_blocks, n_bands, block_size = X_block.shape
    hop_size = block_size - overlap_size
    n_samps = hop_size * n_blocks + overlap_size  # Total length of reconstructed signal

    X = np.zeros([n_samps, n_bands], dtype='complex128')
    for k in np.arange(n_bands):
        # Reconstruct each band independently
        X[:, k] = splicer(X_block[:, k, :], overlap_size)
    return X