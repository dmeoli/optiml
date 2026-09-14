from abc import ABC, abstractmethod

import numpy as np

from .kernels import LinearKernel


class SMO(ABC):
    """
    Base abstract class for the sequential minimal optimization (SMO)
    algorithm used to train the dual SVM formulation.

    The dual of a training problem, whether of a classifier or of a
    regressor, is the same quadratic program over a box with one linear
    equality constraint

        min 1/2 alphas^T Q alphas + q^T alphas ,
            0 <= alphas <= C , s^T alphas = 0 ,

    written over a *dual index space* of size ``n_dual``, of the map giving
    the sample each dual index refers to, of the signs ``s`` and of the
    linear coefficients ``q``. A classifier has one dual index per sample,
    with the sign of its target and a coefficient of -1; a regressor has two,
    one per side of the insensitivity tube, with opposite signs. Everything
    else, this class included, is written in terms of those alone, which is
    why the same iteration trains both.

    The iteration is the one of Platt with the working set selection of
    Keerthi et al. and of Fan, Chen and Lin: the first index of the pair is
    the maximal violating one, the second is the one that decreases the
    objective the most among those that make a violating pair with it. The
    gradient of the dual is maintained, so that both are read off a single
    scan of the active set, and the active set is shrunk of what provably
    cannot be selected any more.

    Subclasses provide the dual index space and read the solution back out of
    the multipliers, and nothing else.
    """

    def __init__(self, quad, X, y, K, kernel, C, tol=1e-3, verbose=False):
        """
        Parameters
        ----------

        quad : `Quadratic` instance
            The quadratic objective of the dual problem, used to monitor
            the cost during the optimization.

        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values associated with ``X``.

        K : ndarray of shape (n_samples, n_samples)
            Precomputed kernel (Gram) matrix of the training data.

        kernel : `Kernel` instance
            The kernel function used to build ``K``. If it is a `LinearKernel`
            the primal weight vector ``w`` is recovered from the multipliers.

        C : float
            Regularization parameter, i.e., the upper bound on the
            Lagrange multipliers.

        tol : float, default=1e-3
            Tolerance for the KKT stopping criterion.

        verbose : bool or int, default=False
            Controls the verbosity of progress messages to stdout.
        """
        self.quad = quad
        self.X = X
        self.y = y
        self.K = K
        self.kernel = kernel
        if isinstance(kernel, LinearKernel):
            self.w = 0.
        self.b = 0.
        self.C = C
        self.tol = tol
        self.iter = 0
        self.verbose = verbose

    # the dual index space - - - - - - - - - - - - - - - - - - - - - - - - - -

    @abstractmethod
    def _dual_index_space(self):
        """
        The dual index space of the training problem, as the triple of the
        sample each dual index refers to, of the signs and of the linear
        coefficients. This is all the iteration knows about the problem.

        Returns
        -------

        smpl : ndarray of shape (n_dual,)
            The sample each dual index refers to.

        s : ndarray of shape (n_dual,)
            The sign of each dual index.

        q : ndarray of shape (n_dual,)
            The linear coefficient of each dual index.
        """

    @abstractmethod
    def _set_solution(self, alphas, smpl, s, m, M):
        """
        Reads the solution the iteration has found back out of the
        multipliers, in the form the caller expects it: the multipliers
        themselves, the bias, and the weight vector when the kernel is linear.

        Parameters
        ----------

        alphas : ndarray of shape (n_dual,)
            The multipliers, in the order of the dual index space.

        smpl : ndarray of shape (n_dual,)
            The sample each dual index refers to.

        s : ndarray of shape (n_dual,)
            The sign of each dual index.

        m, M : float
            The largest and the smallest value of the bias that the
            multipliers allow: at optimality they coincide with it.
        """

    # the iteration - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def minimize(self):
        smpl, s, q = self._dual_index_space()
        n_dual = len(s)
        u = self.C

        # the multipliers and the gradient of the dual at them, G = Q alphas
        # + q, which starts at q since the multipliers start at zero and is
        # then followed in O( n_dual ) at every step rather than recomputed
        # the multipliers of the previous call, if there are any, are a
        # sensible starting point: what makes them one is that the gradient
        # has been followed through the change [see set_C()], so that it is
        # exactly the gradient of the new dual at them, as if it had been
        # recomputed from scratch
        if getattr(self, '_alphas', None) is not None and \
                len(self._alphas) == n_dual:
            alphas = self._alphas.copy()
            G = self._G.copy()
        else:
            alphas = np.zeros(n_dual)
            G = np.asarray(q, dtype=float).copy()

        # the diagonal of the Hessian, which the second order selection reads
        # once per candidate: taking it out of the kernel matrix every time
        # walks that matrix with stride n + 1, i.e. one cache miss per
        # candidate, which is what makes the rule cost more than it saves
        QD = self.K[smpl, smpl].astype(float).copy()

        # the active set is a *prefix* of the current order of the dual
        # indices: shrinking an index is exchanging it with the last active
        # one, which keeps every scan sequential, as it would be without
        # shrinking at all. The kernel matrix is not touched, being indexed by
        # the sample rather than by the dual index, so only the map moves
        smpl = np.asarray(smpl).copy()
        s = np.asarray(s, dtype=float).copy()
        q = np.asarray(q, dtype=float).copy()
        # where each entry of the current order came from, so that the
        # multipliers can be reported in the order the caller has them
        perm = np.arange(n_dual)
        state = (alphas, G, QD, s, q, smpl, perm)
        act = n_dual

        # shrinking at every iteration costs more than it saves, and above all
        # the whole index space has to come back every so often even when
        # nothing asks for it: what makes the active set collapse is not
        # shrinking often but staying shrunk
        period = min(n_dual, 1000)
        counter = period
        patience = 20
        passes = 0
        pm, pM = np.inf, -np.inf

        while True:

            if counter:
                counter -= 1

            if (not counter) and (pM > -np.inf):
                counter = period
                if act < n_dual:
                    passes += 1
                    if passes >= patience:
                        # what the restore is for is that the pair is selected
                        # out of the whole index space again, and one pass at
                        # full width does that: the period is thrown away
                        # together with the interval, so that the shrinking
                        # decides again as soon as a new one is there, a pass
                        # at full width costing O(n_dual) against the O(act)
                        # of every other one
                        act = self._unshrink(state, act, n_dual)
                        passes = 0
                        counter = 0
                        pm, pM = np.inf, -np.inf
                    else:
                        act = self._shrink(state, act, pm, pM)
                else:
                    act = self._shrink(state, act, pm, pM)

            # the first index of the pair: the maximal violating one - - - - -

            sa, aa, Ga = s[:act], alphas[:act], G[:act]
            g = -sa * Ga
            up = np.where(sa > 0, aa < u, aa > 0)
            low = np.where(sa > 0, aa > 0, aa < u)

            if not up.any():
                i, m = -1, -np.inf
            else:
                i = int(np.flatnonzero(up)[np.argmax(g[up])])
                m = g[i]

            M = g[low].min() if low.any() else np.inf

            # the second one: the index of those that make a violating pair
            # with the first that decreases the objective the most, the
            # curvature along the direction that moves the two multipliers
            # being Q_ii + Q_kk - 2 Q_ik

            j = -1
            if i >= 0 and low.any():
                # the curvature along the direction that moves the two
                # multipliers is K_ii + K_kk - 2 K_ik: the signs cancel out,
                # the direction being the one that keeps s^T alphas where it is
                a = QD[i] + QD[:act] - 2 * self.K[smpl[:act], smpl[i]]
                a = np.where(a > 0, a, 1e-12)
                dec = np.where(low & (g < m), (m - g) ** 2 / a, -np.inf)
                if np.isfinite(dec).any():
                    j = int(np.argmax(dec))
                    if not np.isfinite(dec[j]):
                        j = -1

            if i < 0 or j < 0 or m - M <= self.tol:
                if act < n_dual:
                    # the conditions hold on the active set, which says
                    # nothing about the rest: everything comes back and they
                    # are checked where they have to hold
                    act = self._unshrink(state, act, n_dual)
                    passes = 0
                    pm, pM = np.inf, -np.inf
                    continue
                break

            pm, pM = m, M

            # minimize along the only feasible direction changing the two - -

            a = QD[i] + QD[j] - 2 * self.K[smpl[i], smpl[j]]
            if a <= 0:
                a = 1e-12
            t = (m - g[j]) / a
            t = min(t, u - alphas[i] if s[i] > 0 else alphas[i],
                    alphas[j] if s[j] > 0 else u - alphas[j])
            if t <= 0:
                if act < n_dual:
                    act = self._unshrink(state, act, n_dual)
                    passes = 0
                    pm, pM = np.inf, -np.inf
                    continue
                break

            alphas[i] += s[i] * t
            alphas[j] -= s[j] * t

            # snap to the bounds, so that the sets the selection reads off the
            # multipliers are exact
            for k in (i, j):
                if alphas[k] < 1e-12:
                    alphas[k] = 0.
                elif alphas[k] > u - 1e-12:
                    alphas[k] = u

            # the gradient follows in O( n_dual ): this is the innermost loop
            G[:act] += sa * t * (self.K[smpl[:act], smpl[i]] -
                                 self.K[smpl[:act], smpl[j]])

            if self.verbose and not self.iter % self.verbose:
                print('{:4d}\t{: 1.4e}'.format(self.iter, self.quad.function(
                    alphas[np.argsort(state[5] if False else np.arange(n_dual))])))

            self.iter += 1

        # the bias is any value the optimality conditions leave room for,
        # i.e. the midpoint of the interval the multipliers allow
        if not np.isfinite(m):
            m = 0.
        if not np.isfinite(M):
            M = 0.

        # the state of the dual survives the call, in the order the caller
        # has it: the exact solution path of learn()/unlearn() walks it
        order = np.argsort(perm)
        self._alphas = alphas[order].copy()
        self._G = G[order].copy()
        self._s = s[order].copy()
        self._q = q[order].copy()
        self._smpl = smpl[order].copy()
        self._QD = QD[order].copy()
        # the multiplier of the equality constraint, which is what the path
        # walks: the bias the caller sees is what _set_solution() makes of it,
        # and the two do not have the same sign for every problem
        self._b = (m + M) / 2

        self._set_solution(alphas, smpl, s, m, M)

        if self.verbose:
            print()

        return self


    # re-optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def set_C(self, C):
        """
        Changes the trade-off parameter, i.e. the upper bound on the
        multipliers, keeping the solution of the previous call as the starting
        point of the next one.

        The gradient of the dual is affine in the multipliers, in the linear
        term and in the diagonal alike, so the change is followed exactly, and
        in O( n_dual ) time rather than in the O( n_dual^2 ) that recomputing
        it from scratch costs. When the bound *decreases* the multipliers that
        exceed it are *scaled* rather than clipped, which is what keeps them
        feasible for the equality constraint as well, the latter being
        homogeneous, and keeps them in the proportions the previous solve had
        put them in.
        """
        if getattr(self, '_alphas', None) is None:
            self.C = C
            return

        if C < self.C:
            mx = self._alphas.max() if len(self._alphas) else 0.
            if mx > C:
                theta = C / mx
                self._G = theta * (self._G - self._q) + self._q
                self._alphas = self._alphas * theta

        self.C = C

    def set_epsilon(self, epsilon):
        """
        Changes the half-width of the insensitivity tube, which only moves the
        linear term of the dual by the same amount on every dual index, hence
        is followed in O( n_dual ) as well. Meaningless for a classifier,
        whose linear term does not depend on it.
        """
        raise NotImplementedError

    # the exact solution path - - - - - - - - - - - - - - - - - - - - - - - -

    def _dual_indices_of(self, i):
        """
        The dual indices the sample ``i`` carries: one for a classifier, the
        two sides of the tube for a regressor.
        """
        return np.flatnonzero(self._smpl == i)

    def _free_system(self, S, c):
        """
        The matrix of the free system on the margin set @p S, bordered by the
        equality constraint, whose first slot is the border row:

            Q_SS beta + s_S beta_b = - Q_Sc  ,  s_S . beta = - s_c

        Returns the matrix and the right hand side, in that order.
        """
        s, smpl = self._s, self._smpl
        S = np.asarray(S, dtype=int)
        ns = len(S)
        sS, mS = s[S], smpl[S]
        A = np.zeros((ns + 1, ns + 1))
        A[0, 1:] = sS
        A[1:, 0] = sS
        A[1:, 1:] = np.outer(sS, sS) * self.K[np.ix_(mS, mS)]
        rhs = np.empty(ns + 1)
        rhs[0] = -s[c]
        rhs[1:] = -sS * s[c] * self.K[mS, smpl[c]]
        return A, rhs

    def _add_to_free_system(self, R, S, k):
        """
        Extends the inverse of the matrix of the free system, which @p S is
        the margin set of, with the dual index @p k that has just reached the
        margin, by the bordering formula

            gamma = Q_kk - z^T R z  ,  v = R z

        with z the column of @p k against the system as it is: the new
        inverse is R + v v^T / gamma bordered by - v / gamma and 1 / gamma,
        which costs the square of the order of the system rather than its
        cube. Returns None if gamma is too small to divide by, i.e. if the
        extended matrix is singular, in which case the caller computes the
        inverse again from scratch.
        """
        s, smpl = self._s, self._smpl
        S = np.asarray(S, dtype=int)
        z = np.empty(len(S) + 1)
        z[0] = s[k]
        z[1:] = s[S] * s[k] * self.K[smpl[S], smpl[k]]
        v = R.dot(z)
        gamma = self.K[smpl[k], smpl[k]] - z.dot(v)
        if abs(gamma) < 1e-10:
            return None
        n = len(z)
        out = np.empty((n + 1, n + 1))
        out[:n, :n] = R + np.outer(v, v) / gamma
        out[:n, n] = -v / gamma
        out[n, :n] = -v / gamma
        out[n, n] = 1. / gamma
        return out

    @staticmethod
    def _rmv_from_free_system(R, p):
        """
        Drops the slot @p p, i.e. the row and the column of one dual index
        that has left the margin, out of the inverse of the matrix of the free
        system, by the counterpart of the bordering formula

            R_ij - R_ip R_pj / R_pp

        Returns None if R_pp is too small to divide by, in which case the
        caller computes the inverse again from scratch.
        """
        if abs(R[p, p]) < 1e-10:
            return None
        out = R - np.outer(R[:, p], R[p, :]) / R[p, p]
        return np.delete(np.delete(out, p, axis=0), p, axis=1)

    def follow_path(self, c, to, pinned=()):
        """
        Moves the multiplier of the dual index ``c`` towards ``to`` keeping
        *every other* dual index at its own optimality condition, which is the
        incremental and decremental algorithm of Cauwenberghs and Poggio.

        The multipliers of the margin indices and the bias follow the one of
        ``c`` along the direction that the free system gives, up to the first
        *event*: the multiplier of ``c`` arrives, its own condition starts
        holding, a margin multiplier reaches a bound, or a bounded index
        reaches the margin. At each event the direction is recomputed and the
        walk resumes, so what is left behind is optimal at every point of the
        path and exact when it stops. Growing a multiplier from zero learns a
        sample, driving it to zero unlearns one, and the two are the same walk
        taken in opposite directions.

        Since two consecutive events differ by one index, the inverse of the
        matrix of the free system is kept across them and followed with
        rank-one updates, which cost the square of the order of the system
        rather than its cube. Building it costs more than one solve, so a walk
        that ends at its first event never builds it, and it is computed again
        from scratch every so many updates, so that what each of them loses in
        accuracy does not pile up.

        Returns the number of events taken, or -1 if the path could not be
        followed, in which case the multipliers are still feasible but they
        are not the solution asked for and the caller has to fall back on
        ``minimize()``.
        """
        alphas, G, s, q, smpl = (self._alphas, self._G, self._s,
                                 self._q, self._smpl)
        n_dual = len(s)
        u = self.C
        eps = 1e-9
        pinned = set(int(k) for k in pinned) | {int(c)}
        events = 0

        def h(k):
            return G[k] + s[k] * self._b

        S = []
        refactor = True     # the margin set has to be read off the state
        R = None            # the inverse, when it is there and matches S
        updates = 0         # rank-one updates since the last factorization

        for _ in range(20 * n_dual + 100):

            dist = to - alphas[c]
            if abs(dist) <= 1e-12:
                return events
            dirn = 1. if dist > 0 else -1.

            hc = h(c)
            if dirn > 0 and hc >= -1e-9:
                # nothing more is asked of it: its own condition holds where
                # it is
                return events

            if refactor:
                # the margin: the indices whose multiplier is strictly inside
                # its bounds and *also* those that sit at a bound with their
                # own condition holding with equality, which is where a
                # previous event has left them. Reading the set off the
                # multipliers alone makes the walk pick the same event forever
                S = [k for k in range(n_dual)
                     if k not in pinned and
                     ((eps < alphas[k] < u - eps) or abs(h(k)) <= 1e-7)]
                refactor = False
                R = None

            if not S:
                # no multiplier can absorb the movement of c and keep s^T
                # alphas where it is: what moves is the bias alone, until some
                # index stops satisfying its own condition
                bdir = s[c] if hc < 0 else -s[c]
                mag, who = np.inf, -1
                for k in range(n_dual):
                    if k in pinned:
                        continue
                    dh = s[k] * bdir
                    if abs(dh) <= 1e-12:
                        continue
                    t = -h(k) / dh
                    if 1e-12 < t < mag:
                        mag, who = t, k
                tc = -hc / (s[c] * bdir)
                if 0 < tc <= mag:
                    self._b += bdir * tc
                    return events
                if who < 0:
                    return -1
                self._b += bdir * mag
                events += 1
                refactor = True
                continue

            # from the second event on the walk carries the inverse of the
            # system: building it costs more than one solve, hence a walk that
            # ends at its first event never does
            if events and R is None:
                A, _ = self._free_system(S, c)
                try:
                    R = np.linalg.inv(A)
                    updates = 0
                except np.linalg.LinAlgError:
                    R = None

            while True:
                S_arr = np.asarray(S, dtype=int)
                A, rhs = self._free_system(S, c)
                if R is not None:
                    sol = R.dot(rhs)
                else:
                    try:
                        sol = np.linalg.solve(A, rhs)
                    except np.linalg.LinAlgError:
                        return -1
                beta_b, beta = sol[0], sol[1:]

                # whoever is at a bound and would be pushed out of it is not
                # on the margin after all: it leaves and the direction is
                # computed again, which can only happen as many times as there
                # are indices on it
                gone = [idx for idx, k in enumerate(S_arr)
                        if (alphas[k] <= eps and beta[idx] * dirn < -1e-12) or
                        (alphas[k] >= u - eps and beta[idx] * dirn > 1e-12)]
                if not gone:
                    break

                # dropped from the last on, so that the slots before it do not
                # move; the border row takes up the first slot
                if R is not None:
                    for idx in reversed(gone):
                        R = self._rmv_from_free_system(R, idx + 1)
                        if R is None:
                            break
                        updates += 1
                S = [k for idx, k in enumerate(S_arr) if idx not in set(gone)]
                if not S:
                    break

            if not S:
                refactor = True
                continue

            sS, mS = s[S_arr], smpl[S_arr]

            # how the condition of every index moves per unit of movement of c
            w = (s * self.K[smpl, smpl[c]] * s[c] +
                 (s[:, None] * self.K[np.ix_(smpl, mS)] * sS).dot(beta))
            gof = w + s * beta_b

            # the first event along the direction
            mag = abs(dist)
            what, who = 0, -1

            gc = gof[c] * dirn
            if dirn > 0 and gc > 1e-12:
                t = -hc / gc
                if 0 <= t < mag:
                    mag, what = t, 1

            for idx, k in enumerate(S_arr):
                db = beta[idx] * dirn
                if abs(db) <= 1e-12:
                    continue
                room = (u - alphas[k]) if db > 0 else alphas[k]
                t = room / abs(db)
                if t < mag:
                    mag, what, who = t, 2, k

            in_S = set(int(k) for k in S_arr)
            for k in range(n_dual):
                if k in pinned or k in in_S:
                    continue
                dh = gof[k] * dirn
                if abs(dh) <= 1e-12:
                    continue
                t = -h(k) / dh
                if 1e-12 < t < mag:
                    mag, what, who = t, 3, k

            if not np.isfinite(mag):
                return -1

            # walk that far
            da = dirn * mag
            alphas[c] += da
            alphas[S_arr] += beta * da
            self._b += beta_b * da
            G += w * da

            # snap to the bounds, so that the margin is what it looks like
            for k in list(S_arr) + [c] + ([who] if what == 2 else []):
                if alphas[k] < eps:
                    alphas[k] = 0.
                elif alphas[k] > u - eps:
                    alphas[k] = u

            events += 1
            self.iter += 1

            if what == 1:
                return events

            if what == 3:
                # the index that has reached the margin joins the margin set,
                # and the inverse follows it with a rank-one update; the one
                # that has reached a bound stays, its own condition holding
                # there with equality, and leaves only when the direction
                # would push it out, which the pruning above sees to
                if R is not None:
                    R = self._add_to_free_system(R, S, who)
                    updates += 1
                if R is None:
                    refactor = True
                else:
                    S = list(S) + [int(who)]

            # the inverse is computed again from scratch every so many
            # updates, so that what each of them loses in accuracy does not
            # pile up
            if updates >= 50:
                R = None

        return -1

    def unlearn(self, i):
        """
        Drives the multipliers of the sample ``i`` to zero keeping every other
        one at its own optimality condition, so that what is left is the exact
        solution of the training problem *without* that sample, at the cost of
        one walk along the solution path rather than of a training. This is
        what makes the leave-one-out estimate, and the k-fold that unlearns
        one fold at a time, cost a walk each instead of a training each.

        Returns the number of events taken, or -1 if the path could not be
        followed.
        """
        idx = self._dual_indices_of(i)
        events = 0
        for c in idx:
            # the indices of the sample stay pinned for the whole of it: a
            # walk keeps every index that is not pinned at its own condition,
            # and would therefore let one that has already been unlearnt back
            # in, which the problem without the sample has no way of doing
            taken = self.follow_path(int(c), 0., pinned=idx.tolist())
            if taken < 0:
                return -1
            events += taken
        # the sample that has been unlearnt does not take part in the problem
        # any more, hence it does not constrain the bias either
        m, M = self._bias_interval(exclude=idx)
        self._b = (m + M) / 2
        self._set_solution(self._alphas, self._smpl, self._s, m, M)
        return events

    def _bias_interval(self, exclude=()):
        """
        The interval the optimality conditions leave for the bias at the
        current multipliers, whose midpoint is the bias itself. The dual
        indices in @p exclude are the ones of a sample that has been
        unlearnt: they take no part in the problem any more, hence they do
        not constrain the bias either.
        """
        alphas, G, s = self._alphas, self._G, self._s
        u = self.C
        g = -s * G
        up = np.where(s > 0, alphas < u, alphas > 0)
        low = np.where(s > 0, alphas > 0, alphas < u)
        if len(exclude):
            up[np.asarray(exclude, dtype=int)] = False
            low[np.asarray(exclude, dtype=int)] = False
        m = g[up].max() if up.any() else 0.
        M = g[low].min() if low.any() else 0.
        return m, M

    # shrinking - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @staticmethod
    def _swap(state, a, b):
        if a != b:
            for v in state:
                v[a], v[b] = v[b], v[a]

    def _shrink(self, state, act, m, M):
        """
        Takes out of the active set what cannot be selected any more: a
        multiplier that cannot be increased and whose own value of the bias is
        larger than the largest one the others allow can be neither the first
        nor the second of a violating pair, and symmetrically for one that
        cannot be decreased. The exclusion is only valid at the current
        multipliers, which is why everything is put back before the optimality
        conditions are declared to hold.
        """
        alphas, G, QD, s, q, smpl, perm = state
        u = self.C
        k = 0
        while k < act:
            sk = s[k]
            gk = -sk * G[k]
            up = alphas[k] < u if sk > 0 else alphas[k] > 0
            low = alphas[k] > 0 if sk > 0 else alphas[k] < u
            if ((not up) and gk > m) or ((not low) and gk < M):
                act -= 1
                self._swap(state, k, act)
                continue
            k += 1
        return act

    def _unshrink(self, state, act, n_dual):
        """
        Puts back into the active set everything that had been taken out,
        recomputing the gradient of what had been left out, which the steps
        taken in the meantime have made stale: G = Q alphas + q costs one row
        of the kernel matrix per *nonzero* multiplier rather than one per
        index restored, a multiplier that is zero contributing nothing.
        """
        alphas, G, QD, s, q, smpl, perm = state
        if act >= n_dual:
            return n_dual

        nz = np.flatnonzero(alphas)
        G[act:] = q[act:]
        if len(nz):
            G[act:] += s[act:] * (self.K[np.ix_(smpl[act:], smpl[nz])] @
                                  (s[nz] * alphas[nz]))
        return n_dual


class SMOClassifier(SMO):
    """
    The dual index space of a classifier: one dual index per sample, with the
    sign of its target and a linear coefficient of -1.
    """

    def __init__(self, quad, X, y, K, kernel, C, tol=1e-3, verbose=False):
        self.alphas = np.zeros(len(X))
        super(SMOClassifier, self).__init__(quad, X, y, K, kernel, C, tol, verbose)

    def _dual_index_space(self):
        n_samples = len(self.X)
        return (np.arange(n_samples),
                np.asarray(self.y, dtype=float),
                -np.ones(n_samples))

    def _set_solution(self, alphas, smpl, s, m, M):
        self.alphas = np.zeros(len(self.X))
        self.alphas[smpl] = alphas
        self.b = (m + M) / 2
        if isinstance(self.kernel, LinearKernel):
            self.w = (s * alphas).dot(self.X[smpl])


class SMORegression(SMO):
    """
    The dual index space of a regressor: two dual indices per sample, one per
    side of the insensitivity tube, with opposite signs and the target of the
    sample shifted by the half-width of the tube.
    """

    def __init__(self, quad, X, y, K, kernel, C, epsilon, tol=1e-3, verbose=False):
        self.alphas_p = np.zeros(len(X))
        self.alphas_n = np.zeros(len(X))
        self.epsilon = epsilon
        super(SMORegression, self).__init__(quad, X, y, K, kernel, C, tol, verbose)

    def _dual_index_space(self):
        n_samples = len(self.X)
        smpl = np.concatenate((np.arange(n_samples), np.arange(n_samples)))
        s = np.concatenate((np.ones(n_samples), -np.ones(n_samples)))
        q = np.concatenate((-self.y, self.y)) + self.epsilon
        return smpl, s, np.asarray(q, dtype=float)

    def set_epsilon(self, epsilon):
        if getattr(self, '_q', None) is not None:
            self._G += epsilon - self.epsilon
            self._q += epsilon - self.epsilon
        self.epsilon = epsilon

    def _set_solution(self, alphas, smpl, s, m, M):
        n_samples = len(self.X)
        full = np.zeros(2 * n_samples)
        # the two sides of the tube are told apart by the sign, the sample
        # alone not saying which of the two dual indices one is
        full[np.where(s > 0, smpl, smpl + n_samples).astype(int)] = alphas
        self.alphas_p = full[:n_samples]
        self.alphas_n = full[n_samples:]
        self.b = -(m + M) / 2
        if isinstance(self.kernel, LinearKernel):
            self.w = -(self.alphas_p - self.alphas_n).dot(self.X)
