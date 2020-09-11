import numpy as np
import warnings


def autocorrelate(a, m=16, deltat=1, normalize=False,
                  copy=True, dtype=None):
    """
    Autocorrelation of a 1-dimensional sequence on a log2-scale.

    This computes the correlation similar to
    :py:func:`numpy.correlate` for positive :math:`k` on a base 2
    logarithmic scale.

        :func:`numpy.correlate(a, a, mode="full")[len(a)-1:]`

        :math:`z_k = \Sigma_n a_n a_{n+k}`


    Parameters
    ----------
    a : array-like
        input sequence
    m : even integer
        defines the number of points on one level, must be an
        even integer
    deltat : float
        distance between bins
    normalize : bool
        True: normalize the result to the square of the average input
        signal and the factor :math:`M-k`.
        False: normalize the result to the factor :math:`M-k`.
    copy : bool
        copy input array, set to ``False`` to save memory
    dtype : object to be converted to a data type object
        The data type of the returned array and of the accumulator
        for the multiple-tau computation.


    Returns
    -------
    autocorrelation : ndarray of shape (N,2)
        the lag time (1st column) and the autocorrelation (2nd column).

    Notes
    -----
    .. versionchanged :: 0.1.6
       Compute the correlation for zero lag time.

    The algorithm computes the correlation with the convention of the
    curve decaying to zero.

    For experiments like e.g. fluorescence correlation spectroscopy,
    the signal can be normalized to :math:`M-k`
    by invoking ``normalize = True``.

    For normalizing according to the behavior
    of :py:func:`numpy.correlate`, use ``normalize = False``.

    Examples
    --------
    >>> from multipletau import autocorrelate
    >>> autocorrelate(range(42), m=2, dtype=np.float_)
    array([[  0.00000000e+00,   2.38210000e+04],
           [  1.00000000e+00,   2.29600000e+04],
           [  2.00000000e+00,   2.21000000e+04],
           [  4.00000000e+00,   2.03775000e+04],
           [  8.00000000e+00,   1.50612000e+04]])
    """
    assert isinstance(copy, bool)
    assert isinstance(normalize, bool)

    if dtype is None:
        dtype = np.dtype(a[0].__class__)
    else:
        dtype = np.dtype(dtype)


    if dtype.kind != "f":
        warnings.warn("Input dtype is not float; casting to np.float_!")
        dtype = np.dtype(np.float_)

    # If copy is false and dtype is the same as the input array,
    # then this line does not have an effect:
    trace = np.array(a, dtype=dtype, copy=copy)

    # Check parameters
    if m // 2 != m / 2:
        mold = m
        m = np.int_((m // 2 + 1) * 2)
        warnings.warn("Invalid value of m={}. Using m={} instead"
                      .format(mold, m))
    else:
        m = np.int_(m)

    N = N0 = trace.shape[0]

    # Find out the length of the correlation function.
    # The integer k defines how many times we can average over
    # two neighboring array elements in order to obtain an array of
    # length just larger than m.
    k = np.int_(np.floor(np.log2(N / m)))

    # In the base2 multiple-tau scheme, the length of the correlation
    # array is (only taking into account values that are computed from
    # traces that are just larger than m):
    lenG = m + k * (m // 2) + 1

    G = np.zeros((lenG, 2), dtype=dtype)

    normstat = np.zeros(lenG, dtype=dtype)
    normnump = np.zeros(lenG, dtype=dtype)

    traceavg = np.average(trace)

    # We use the fluctuation of the signal around the mean
    if normalize:
        # trace -= traceavg #Not interested in signal around mean, correlation should drop to 1 when normalized!!
        # assert traceavg != 0, "Cannot normalize: Average of `a` is zero!"
        if traceavg ==0: traceavg=1

    # Otherwise the following for-loop will fail:
    assert N >= 2 * m, "len(a) must be larger than 2m!"

    # Calculate autocorrelation function for first m+1 bins
    # Discrete convolution of m elements
    for n in range(0, m + 1):
        G[n, 0] = deltat * n
        # This is the computationally intensive step
        G[n, 1] = np.sum(trace[:N - n] * trace[n:])
        normstat[n] = N - n
        normnump[n] = N
    # Now that we calculated the first m elements of G, let us
    # go on with the next m/2 elements.
    # Check if len(trace) is even:
    if N % 2 == 1:
        N -= 1
    # Add up every second element
    trace = (trace[:N:2] + trace[1:N:2]) / 2
    N //= 2
    # Start iteration for each m/2 values
    for step in range(1, k + 1):
        # Get the next m/2 values via correlation of the trace
        for n in range(1, m // 2 + 1):
            npmd2 = n + m // 2
            idx = m + n + (step - 1) * m // 2
            if len(trace[:N - npmd2]) == 0:
                # This is a shortcut that stops the iteration once the
                # length of the trace is too small to compute a corre-
                # lation. The actual length of the correlation function
                # does not only depend on k - We also must be able to
                # perform the sum with respect to k for all elements.
                # For small N, the sum over zero elements would be
                # computed here.
                #
                # One could make this for-loop go up to maxval, where
                #   maxval1 = int(m/2)
                #   maxval2 = int(N-m/2-1)
                #   maxval = min(maxval1, maxval2)
                # However, we then would also need to find out which
                # element in G is the last element...
                G = G[:idx - 1]
                normstat = normstat[:idx - 1]
                normnump = normnump[:idx - 1]
                # Note that this break only breaks out of the current
                # for loop. However, we are already in the last loop
                # of the step-for-loop. That is because we calculated
                # k in advance.
                break
            else:
                G[idx, 0] = deltat * npmd2 * 2**step
                # This is the computationally intensive step
                G[idx, 1] = np.sum(trace[:N - npmd2] *
                                   trace[npmd2:])
                normstat[idx] = N - npmd2
                normnump[idx] = N
        # Check if len(trace) is even:
        if N % 2 == 1:
            N -= 1
        # Add up every second element
        trace = (trace[:N:2] + trace[1:N:2]) / 2
        N //= 2

    if normalize:
        G[:, 1] /= traceavg**2 * normstat
    else:
        G[:, 1] /= normstat

    return G