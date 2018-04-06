"""
Implements the Kalman filter for a linear Gaussian state space model.

References
----------

https://lectures.quantecon.org/py/kalman.html

"""
from textwrap import dedent
import numpy as np
from scipy.linalg import inv
from quantecon.lss import LinearStateSpace
from quantecon.matrix_eqn import solve_discrete_riccati


class Kalman:
    r"""
    Implements the Kalman filter for the Gaussian state space model

    .. math::

        x_{t+1} = A x_t + C w_{t+1} \\
        y_t = G x_t + H v_t

    Here :math:`x_t` is the hidden state and :math:`y_t` is the measurement.
    The shocks :math:`w_t` and :math:`v_t` are iid standard normals. Below
    we use the notation

    .. math::

        Q := CC'
        R := HH'


    Parameters
    -----------
    ss : instance of LinearStateSpace
        An instance of the quantecon.lss.LinearStateSpace class
    x_hat : scalar(float) or array_like(float), optional(default=None)
        An n x 1 array representing the mean x_hat and covariance
        matrix Sigma of the prior/predictive density.  Set to zero if
        not supplied.
    Sigma : scalar(float) or array_like(float), optional(default=None)
        An n x n array representing the covariance matrix Sigma of
        the prior/predictive density.  Must be positive definite.
        Set to the identity if not supplied.

    Attributes
    ----------
    Sigma, x_hat : as above
    Sigma_infinity : array_like or scalar(float)
        The infinite limit of Sigma_t
    K_infinity : array_like or scalar(float)
        The stationary Kalman gain.


    References
    ----------

    https://lectures.quantecon.org/py/kalman.html

    """

    def __init__(self, ss, x_hat=None, Sigma=None):
        self.ss = ss
        self.set_state(x_hat, Sigma)
        self._K_infinity = None
        self._Sigma_infinity = None

    def set_state(self, x_hat, Sigma):
        if Sigma is None:
            Sigma = np.identity(self.ss.n)
        else:
            self.Sigma = np.atleast_2d(Sigma)
        if x_hat is None:
            x_hat = np.zeros((self.ss.n, 1))
        else:
            self.x_hat = np.atleast_2d(x_hat)
            self.x_hat.shape = self.ss.n, 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = """\
        Kalman filter:
          - dimension of state space          : {n}
          - dimension of observation equation : {k}
        """
        return dedent(m.format(n=self.ss.n, k=self.ss.k))

    @property
    def Sigma_infinity(self):
        if self._Sigma_infinity is None:
            self.stationary_values()
        return self._Sigma_infinity

    @property
    def K_infinity(self):
        if self._K_infinity is None:
            self.stationary_values()
        return self._K_infinity

    def whitener_lss(self):
        r"""
        This function takes the linear state space system
        that is an input to the Kalman class and it converts
        that system to the time-invariant whitener represenation
        given by

        .. math::

            \tilde{x}_{t+1}^* = \tilde{A} \tilde{x} + \tilde{C} v
            a = \tilde{G} \tilde{x}

        where

        .. math::

            \tilde{x}_t = [x+{t}, \hat{x}_{t}, v_{t}]

        and

        .. math::

            \tilde{A} =
            \begin{bmatrix}
            A  & 0    & 0  \\
            KG & A-KG & KH \\
            0  & 0    & 0 \\
            \end{bmatrix}

        .. math::

            \tilde{C} =
            \begin{bmatrix}
            C & 0 \\
            0 & 0 \\
            0 & I \\
            \end{bmatrix}

        .. math::

            \tilde{G} =
            \begin{bmatrix}
            G & -G & H \\
            \end{bmatrix}

        with :math:`A, C, G, H` coming from the linear state space system
        that defines the Kalman instance

        Returns
        -------
        whitened_lss : LinearStateSpace
            This is the linear state space system that represents
            the whitened system
        """
        K = self.K_infinity

        # Get the matrix sizes
        n, k, m, l = self.ss.n, self.ss.k, self.ss.m, self.ss.l
        A, C, G, H = self.ss.A, self.ss.C, self.ss.G, self.ss.H

        Atil = np.vstack([np.hstack([A, np.zeros((n, n)), np.zeros((n, l))]),
                          np.hstack([K @ G, A-K @ G, K @ H]),
                          np.zeros((l, 2*n + l))])

        Ctil = np.vstack([np.hstack([C, np.zeros((n, l))]),
                          np.zeros((n, m+l)),
                          np.hstack([np.zeros((l, m)), np.eye(l)])])

        Gtil = np.hstack([G, -G, H])

        whitened_lss = LinearStateSpace(Atil, Ctil, Gtil)
        self.whitened_lss = whitened_lss

        return whitened_lss

    def prior_to_filtered(self, y):
        r"""
        Updates the moments (x_hat, Sigma) of the time t prior to the
        time t filtering distribution, using current measurement :math:`y_t`.

        The updates are according to

        .. math::

            \hat{x}^F = \hat{x} + \Sigma G' (G \Sigma G' + R)^{-1}
                (y - G \hat{x})
            \Sigma^F = \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G
                \Sigma

        Parameters
        ----------
        y : scalar or array_like(float)
            The current measurement

        """
        # === simplify notation === #
        G, H = self.ss.G, self.ss.H
        R = H @ H.T

        # === and then update === #
        y = np.atleast_2d(y)
        y.shape = self.ss.k, 1
        E = self.Sigma @ G.T
        F = G @ self.Sigma @ G.T + R
        M = E @ inv(F)
        self.x_hat = self.x_hat + M @ (y - G @ self.x_hat)
        self.Sigma = self.Sigma - M @ (G @ self.Sigma)

    def filtered_to_forecast(self):
        """
        Updates the moments of the time t filtering distribution to the
        moments of the predictive distribution, which becomes the time
        t+1 prior

        """
        # === simplify notation === #
        A, C = self.ss.A, self.ss.C
        Q = C @ C.T

        # === and then update === #
        self.x_hat = A @ self.x_hat
        self.Sigma = A @ self.Sigma @ A.T + Q

    def update(self, y):
        """
        Updates x_hat and Sigma given k x 1 ndarray y.  The full
        update, from one period to the next

        Parameters
        ----------
        y : np.ndarray
            A k x 1 ndarray y representing the current measurement

        """
        self.prior_to_filtered(y)
        self.filtered_to_forecast()

    def stationary_values(self):
        """
        Computes the limit of :math:`\Sigma_t` as t goes to infinity by
        solving the associated Riccati equation. Computation is via the
        doubling algorithm (see the documentation in
        `matrix_eqn.solve_discrete_riccati`).

        Returns
        -------
        Sigma_infinity : array_like or scalar(float)
            The infinite limit of :math:`\Sigma_t`
        K_infinity : array_like or scalar(float)
            The stationary Kalman gain.

        """

        # === simplify notation === #
        A, C, G, H = self.ss.A, self.ss.C, self.ss.G, self.ss.H
        Q, R = C @ C.T, H @ H.T

        # === solve Riccati equation, obtain Kalman gain === #
        Sigma_infinity = solve_discrete_riccati(A.T, G.T, Q, R)
        temp1 = A @ Sigma_infinity @ G.T
        temp2 = inv(G @ Sigma_infinity @ G.T + R)
        K_infinity = temp1 @ temp2

        # == record as attributes and return == #
        self._Sigma_infinity, self._K_infinity = Sigma_infinity, K_infinity
        return Sigma_infinity, K_infinity

    def stationary_coefficients(self, j, coeff_type='ma'):
        """
        Wold representation moving average or VAR coefficients for the
        steady state Kalman filter.

        Parameters
        ----------
        j : int
            The lag length
        coeff_type : string, either 'ma' or 'var' (default='ma')
            The type of coefficent sequence to compute.  Either 'ma' for
            moving average or 'var' for VAR.
        """
        # == simplify notation == #
        A, G = self.ss.A, self.ss.G
        K_infinity = self.K_infinity
        # == compute and return coefficients == #
        coeffs = []
        i = 1
        if coeff_type == 'ma':
            coeffs.append(np.identity(self.ss.k))
            P_mat = A
            P = np.identity(self.ss.n)  # Create a copy
        elif coeff_type == 'var':
            coeffs.append(G @ K_infinity)
            P_mat = A - K_infinity @ G
            P = np.copy(P_mat)  # Create a copy
        else:
            raise ValueError("Unknown coefficient type")
        while i <= j:
            coeffs.append(G @ P @ K_infinity)
            P = P @ P_mat
            i += 1
        return coeffs

    def stationary_innovation_covar(self):
        # == simplify notation == #
        H, G = self.ss.H, self.ss.G
        R = H @ H.T
        Sigma_infinity = self.Sigma_infinity

        return G @ Sigma_infinity @ G.T + R


@jit(nopython=True)
def filter_estimation(A, C, G, H, x_hat, Σ, y):
    """
    Estimates the linear state space model defined by `A`, `C`, `G` and `H` by
    computing the likelihood and estimating the state and variance.
    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    A : ndarray(float)
        See `Kalman`.
    C : ndarray(float)
        See `Kalman`.
    G : ndarray(float)
        See `Kalman`.
    H : ndarray(float)
        See `Kalman`.
    x_hat : ndarray(float)
        An n x 1 array representing the mean x_hat and covariance
        matrix Σ of the prior/predictive density.
    Σ : ndarray(float)
        An n x n array representing the covariance matrix Sigma of
        the prior/predictive density. Must be positive definite.
    y : ndarray(float, ndim=(nrows, ncols))
        Numpy array containing the data values.

    Returns
    ----------
    log_likelihood : ndarray(float, ndim=(nrows, 1))
        Array contaning the log likelihood for each row of observations.
    x_hats : ndarray(float, ndim=(x_hat.shape[0], x_hat.shape[1], nrows))
        Array containing the estimated hidden states.
    Σs : ndarray(float, ndim=(Σ.shape[0], Σ.shape[1], nrows))
        Array containing the estimated variances.

    """
    nrows = y.shape[0]

    # Pre-allocate
    log_likelihood = np.zeros((nrows, 1))
    x_hats = np.zeros((x_hat.shape[0], x_hat.shape[1], nrows))
    Σs = np.zeros((Σ.shape[0], Σ.shape[1], nrows))
    for t in range(nrows):
        # Store state estimate for period t
        x_hats[:, :, t] = x_hat
        Σs[:, :, t] = Σ

        # Adjust the linear state space model for missing observations
        observations_location = ~np.isnan(y[t])
        nb_obs = observations_location.sum()

        _G = G[observations_location, :]
        _H = H[observations_location, :]

        # Compute the likelihood for period t
        E = Σ @ _G.T
        F = _G @ E + _H @ _H.T
        F_inv = inv(F)

        M = E @ F_inv

        mean = _G @ x_hat
        demeaned = y[t][observations_location].reshape((nb_obs, 1)) - mean

        log_det_cov = np.log(det(F))

        log_likelihood[t] = (
            -nb_obs * np.log(2 * np.pi) -
            (1/2) * (log_det_cov + demeaned.T @ F_inv @ demeaned)
        )

        x_hat = A @ (x_hat + M @ demeaned)
        Σ = A @ (Σ - E @ F_inv.T @ E.T) @ A.T + C @ C.T

    return (log_likelihood, x_hats, Σs)
