import sys
import warnings
from abc import ABC

import numpy as np
from sklearn.exceptions import PositiveSpectrumWarning

from .kernels import LinearKernel


class SMO(ABC):

    def __init__(self, quad, X, y, K, kernel, C, tol=1e-3, verbose=False):
        self.quad = quad
        self.X = X
        self.y = y
        self.K = K
        self.kernel = kernel
        if isinstance(kernel, LinearKernel):
            self.w = 0.
        self.b = 0.
        self.C = C
        self.errors = np.zeros(len(X))
        self.tol = tol
        self.iter = 0
        self.verbose = verbose

    def _take_step(self, i1, i2):
        raise NotImplementedError

    def _examine_example(self, i2):
        raise NotImplementedError

    def minimize(self):
        raise NotImplementedError


class SMOClassifier(SMO):
    """
    Implements John Platt's sequential minimal optimization
    algorithm for training a support vector classifier.

    The SMO algorithm is an algorithm for solving large quadratic programming (QP)
    optimization problems, widely used for the training of support vector machines.
    First developed by John C. Platt in 1998, SMO breaks up large QP problems into a
    series of smallest possible QP problems, which are then solved analytically.

    This class follows the original algorithm by Platt with additional modifications
    by Keerthi et al.

    References
    ----------

    John C. Platt. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines.

    S.S. Keerthi, S.K. Shevade, C. Bhattacharyya, K.R.K. Murthy. Improvements to Platt's SMO
    Algorithm for SVM Classifier Design. Technical Report CD-99-14.
    """

    def __init__(self, quad, X, y, K, kernel, C, tol=1e-3, verbose=False):
        self.alphas = np.zeros(len(X))
        super(SMOClassifier, self).__init__(quad, X, y, K, kernel, C, tol, verbose)

        # initialize variables and structures to implement improvements
        # on the original Platt's SMO algorithm described in Keerthi et
        # al. for better performance ed efficiency

        # set of indices
        # {i : 0 < alphas[i] < C}
        self.I0 = set()
        # {i : y[i] = +1, alphas[i] = 0}
        self.I1 = set(i for i in range(len(X)) if y[i] == 1)
        # {i : y[i] = -1, alphas[i] = C}
        self.I2 = set()
        # {i : y[i] = +1, alphas[i] = C}
        self.I3 = set()
        # {i : y[i] = -1, alphas[i] = 0}
        self.I4 = set(i for i in range(len(X)) if y[i] == -1)

        # multiple thresholds
        self.b_up = -1
        self.b_low = 1
        # initialize b_up_idx to any one index of class +1
        self.b_up_idx = next(i for i in range(len(X)) if y[i] == 1)
        # initialize b_low_idx to any one index of class -1
        self.b_low_idx = next(i for i in range(len(X)) if y[i] == -1)

        self.errors[self.b_up_idx] = -1
        self.errors[self.b_low_idx] = 1

    def _take_step(self, i1, i2):
        # skip if chosen alphas are the same
        if i1 == i2:
            return False

        alpha1 = self.alphas[i1]
        y1 = self.y[i1]
        E1 = self.errors[i1]

        alpha2 = self.alphas[i2]
        y2 = self.y[i2]
        E2 = self.errors[i2]

        s = y1 * y2

        # compute L and H, the bounds on new possible alpha values
        # based on equations 13 and 14 in Platt's paper
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return False

        # compute the 2nd derivative of the objective function along
        # the diagonal line based on equation 15 in Platt's paper
        eta = self.K[i1, i1] + self.K[i2, i2] - 2 * self.K[i1, i2]

        # under normal circumstances, the objective function will be positive
        # definite, there will be a minimum along the direction of the linear
        # equality constraint, and eta will be greater than zero compute new
        # alpha2, a2, if eta is positive based on equation 16 in Platt's paper
        if eta > 0:
            # clip a2 based on bounds L and H based
            # on equation 17 in Platt's paper
            a2 = max(L, min(alpha2 + y2 * (E1 - E2) / eta, H))
        else:
            Lobj = y2 * (E1 - E2) * L
            Hobj = y2 * (E1 - E2) * H

            if Lobj > Hobj + 1e-12:
                a2 = L
            elif Lobj < Hobj - 1e-12:
                a2 = H
            else:
                a2 = alpha2

            warnings.warn('kernel matrix is not positive definite', PositiveSpectrumWarning)

        # if examples can't be optimized within tol, skip this pair
        if abs(a2 - alpha2) < 1e-12 * (a2 + alpha2 + 1e-12):
            return False

        # calculate new alpha1 based on equation 18 in Platt's paper
        a1 = alpha1 + s * (alpha2 - a2)

        # update weight vector to reflect change in a1 and a2, if
        # kernel is linear, based on equation 22 in Platt's paper
        if self.kernel == 'linear':
            self.w += y1 * (a1 - alpha1) * self.X[i1] + y2 * (a2 - alpha2) * self.X[i2]

        # update error cache using new alphas
        for i in self.I0:
            if i != i1 and i != i2:
                self.errors[i] += y1 * (a1 - alpha1) * self.K[i1, i] + y2 * (a2 - alpha2) * self.K[i2, i]
        # update error cache using new alphas for i1 and i2
        self.errors[i1] += y1 * (a1 - alpha1) * self.K[i1, i1] + y2 * (a2 - alpha2) * self.K[i1, i2]
        self.errors[i2] += y1 * (a1 - alpha1) * self.K[i1, i2] + y2 * (a2 - alpha2) * self.K[i2, i2]

        # to prevent precision problems
        if a2 > self.C - 1e-8 * self.C:
            a2 = self.C
        elif a2 <= 1e-8 * self.C:
            a2 = 0.

        if a1 > self.C - 1e-8 * self.C:
            a1 = self.C
        elif a1 <= 1e-8 * self.C:
            a1 = 0.

        # update model object with new alphas
        self.alphas[i1] = a1
        self.alphas[i2] = a2

        # update the sets of indices for i1 and i2
        for i in (i1, i2):
            if 0 < self.alphas[i] < self.C:
                self.I0.add(i)
            else:
                self.I0.discard(i)
            if self.y[i] == 1 and self.alphas[i] == 0:
                self.I1.add(i)
            else:
                self.I1.discard(i)
            if self.y[i] == -1 and self.alphas[i] == self.C:
                self.I2.add(i)
            else:
                self.I2.discard(i)
            if self.y[i] == 1 and self.alphas[i] == self.C:
                self.I3.add(i)
            else:
                self.I3.discard(i)
            if self.y[i] == -1 and self.alphas[i] == 0:
                self.I4.add(i)
            else:
                self.I4.discard(i)

        # update thresholds (b_up, b_up_idx) and (b_low, b_low_idx)
        # by applying equations 11a and 11b, using only i1, i2 and
        # indices in I0 as suggested in item 3 of section 5 in
        # Keerthi et al.
        self.b_up_idx = -1
        self.b_low_idx = -1
        self.b_up = sys.float_info.max
        self.b_low = -sys.float_info.max

        for i in self.I0:
            if self.errors[i] > self.b_low:
                self.b_low = self.errors[i]
                self.b_low_idx = i
            if self.errors[i] < self.b_up:
                self.b_up = self.errors[i]
                self.b_up_idx = i
        if i1 not in self.I0:
            if i1 in self.I3 or i1 in self.I4:
                if self.errors[i1] > self.b_low:
                    self.b_low = self.errors[i1]
                    self.b_low_idx = i1
            elif self.errors[i1] < self.b_up:
                self.b_up = self.errors[i1]
                self.b_up_idx = i1
        if i2 not in self.I0:
            if i2 in self.I3 or i2 in self.I4:
                if self.errors[i2] > self.b_low:
                    self.b_low = self.errors[i2]
                    self.b_low_idx = i2
            elif self.errors[i2] < self.b_up:
                self.b_up = self.errors[i2]
                self.b_up_idx = i2

        if self.b_low_idx == -1 or self.b_up_idx == -1:
            raise Exception('unexpected status')

        return True

    def _examine_example(self, i2):
        if i2 in self.I0:
            E2 = self.errors[i2]
        else:
            E2 = (self.alphas * self.y).dot(self.K[i2]) - self.y[i2]
            self.errors[i2] = E2

            # update (b_up, b_up_idx) or (b_low, b_low_idx) using E2 and i2
            if (i2 in self.I1 or i2 in self.I2) and E2 < self.b_up:
                self.b_up = E2
                self.b_up_idx = i2
            elif (i2 in self.I3 or i2 in self.I4) and E2 > self.b_low:
                self.b_low = E2
                self.b_low_idx = i2

        # check optimality using current b_up and b_low and, if violated,
        # find another index i1 to do joint optimization with i2
        i1 = -1
        optimal = True
        if i2 in self.I0 or i2 in self.I1 or i2 in self.I2:
            if self.b_low - E2 > 2 * self.tol:
                optimal = False
                i1 = self.b_low_idx
        if i2 in self.I0 or i2 in self.I3 or i2 in self.I4:
            if E2 - self.b_up > 2 * self.tol:
                optimal = False
                i1 = self.b_up_idx

        if optimal:
            return False

        # for i2 in I0 choose the better i1
        if i2 in self.I0:
            if self.b_low - E2 > E2 - self.b_up:
                i1 = self.b_low_idx
            else:
                i1 = self.b_up_idx

        if i1 == -1:
            raise Exception('the index could not be found')

        return self._take_step(i1, i2)

    def minimize(self):
        if self.verbose:
            print('iter\t cost')

        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            # loop over all training examples
            if examine_all:
                for i in range(len(self.X)):
                    num_changed += self._examine_example(i)
            else:
                # loop over examples where alphas are not already at their limits
                for i in range(len(self.X)):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i)
                        # check if optimality on I0 is attained
                        if self.b_up > self.b_low - 2 * self.tol:
                            num_changed = 0
                            break
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            if self.verbose and not self.iter % self.verbose:
                print('{:4d}\t{: 1.4e}'.format(self.iter, self.quad.function(self.alphas)))

            self.iter += 1

        self.b = -(self.b_low + self.b_up) / 2

        if self.verbose:
            print()

        return self


class SMORegression(SMO):
    """
    Implements Smola and Scholkopf sequential minimal optimization
    algorithm for training a support vector regression.

    The SMO algorithm is an algorithm for solving large quadratic programming (QP)
    optimization problems, widely used for the training of support vector machines.
    First developed by John C. Platt in 1998, SMO breaks up large QP problems into a
    series of smallest possible QP problems, which are then solved analytically.

    This class incorporates modifications in the original SMO algorithm to solve
    regression problems as suggested by Alex J. Smola and Bernhard Scholkopf and
    further modifications for better performance by Shevade et al.

    References
    ----------

    G.W. Flake, S. Lawrence. Efficient SVM Regression Training with SMO.

    Alex J. Smola, Bernhard Scholkopf. A Tutorial on Support Vector Regression.
    NeuroCOLT2 Technical Report Series NC2-TR-1998-030.

    S.K. Shevade, S.S. Keerthi, C. Bhattacharyya, K.R.K. Murthy. Improvements to SMO
    Algorithm for SVM Regression. Technical Report CD-99-16.
    """

    def __init__(self, quad, X, y, K, kernel, C, epsilon, tol=1e-3, verbose=False):
        self.alphas_p = np.zeros(len(X))
        self.alphas_n = np.zeros(len(X))
        super(SMORegression, self).__init__(quad, X, y, K, kernel, C, tol, verbose)
        self.epsilon = epsilon

        # initialize variables and structures to implement improvements
        # on the original Smola and Scholkopf SMO algorithm described in
        # Shevade et al. for better performance ed efficiency

        # set of indices
        # {i : 0 < alphas_p[i] < C, 0 < alphas_n[i] < C}
        self.I0 = set()
        # {i : alphas_p[i] = 0, alphas_n[i] = 0}
        self.I1 = set(range(len(X)))
        # {i : alphas_p[i] = 0, alphas_n[i] = C}
        self.I2 = set()
        # {i : alphas_p[i] = C, alphas_n[i] = 0}
        self.I3 = set()

        # multiple thresholds
        self.b_up_idx = 0
        self.b_low_idx = 0
        self.b_up = y[self.b_up_idx] + self.epsilon
        self.b_low = y[self.b_low_idx] - self.epsilon

    def _take_step(self, i1, i2):
        # skip if chosen alphas are the same
        if i1 == i2:
            return False

        alpha1_p, alpha1_n = self.alphas_p[i1], self.alphas_n[i1]
        E1 = self.errors[i1]

        alpha2_p, alpha2_n = self.alphas_p[i2], self.alphas_n[i2]
        E2 = self.errors[i2]

        # compute kernel and 2nd derivative eta
        # based on equation 15 in Platt's paper
        eta = self.K[i1, i1] + self.K[i2, i2] - 2 * self.K[i1, i2]

        if eta < 0:
            eta = 0

        gamma = alpha1_p - alpha1_n + alpha2_p - alpha2_n

        case1 = case2 = case3 = case4 = False
        changed = finished = False

        delta_E = E1 - E2

        while not finished:  # occurs at most three times
            if (not case1 and
                    (alpha1_p > 0 or (alpha1_n == 0 and delta_E > 0)) and
                    (alpha2_p > 0 or (alpha2_n == 0 and delta_E < 0))):
                # compute L and H wrt alpha1_p, alpha2_p
                L = max(0, gamma - self.C)
                H = min(self.C, gamma)
                if L < H:
                    if eta > 0:
                        a2 = max(L, min(alpha2_p - delta_E / eta, H))
                    else:
                        Lobj = -L * delta_E
                        Hobj = -H * delta_E
                        a2 = L if Lobj > Hobj else H
                        warnings.warn('kernel matrix is not positive definite', PositiveSpectrumWarning)
                    a1 = alpha1_p - (a2 - alpha2_p)
                    # update alpha1, alpha2_p if change is larger than some eps
                    if abs(a1 - alpha1_p) > 1e-12 or abs(a2 - alpha2_p) > 1e-12:
                        alpha1_p = a1
                        alpha2_p = a2
                        changed = True
                else:
                    finished = True
                case1 = True
            elif (not case2 and
                  (alpha1_p > 0 or (alpha1_n == 0 and delta_E > 2 * self.epsilon)) and
                  (alpha2_n > 0 or (alpha2_p == 0 and delta_E > 2 * self.epsilon))):
                # compute L and H wrt alpha1_p, alpha2_n
                L = max(0, -gamma)
                H = min(self.C, -gamma + self.C)
                if L < H:
                    if eta > 0:
                        a2 = max(L, min(alpha2_n + (delta_E - 2 * self.epsilon) / eta, H))
                    else:
                        Lobj = L * (-2 * self.epsilon + delta_E)
                        Hobj = H * (-2 * self.epsilon + delta_E)
                        a2 = L if Lobj > Hobj else H
                        warnings.warn('kernel matrix is not positive definite', PositiveSpectrumWarning)
                    a1 = alpha1_p + (a2 - alpha2_n)
                    # update alpha1, alpha2_n if change is larger than some eps
                    if abs(a1 - alpha1_p) > 1e-12 or abs(a2 - alpha2_n) > 1e-12:
                        alpha1_p = a1
                        alpha2_n = a2
                        changed = True
                else:
                    finished = True
                case2 = True
            elif (not case3 and
                  (alpha1_n > 0 or (alpha1_p == 0 and delta_E < -2 * self.epsilon)) and
                  (alpha2_p > 0 or (alpha2_n == 0 and delta_E < -2 * self.epsilon))):
                # computer L and H wrt alpha1_n, alpha2_p
                L = max(0, gamma)
                H = min(self.C, self.C + gamma)
                if L < H:
                    if eta > 0:
                        a2 = max(L, min(alpha2_p - (delta_E + 2 * self.epsilon) / eta, H))
                    else:
                        Lobj = -L * (2 * self.epsilon + delta_E)
                        Hobj = -H * (2 * self.epsilon + delta_E)
                        a2 = L if Lobj > Hobj else H
                        warnings.warn('kernel matrix is not positive definite', PositiveSpectrumWarning)
                    a1 = alpha1_n + (a2 - alpha2_p)
                    # update alpha1_n, alpha2_p if change is larger than some eps
                    if abs(a1 - alpha1_n) > 1e-12 or abs(a2 - alpha2_p) > 1e-12:
                        alpha1_n = a1
                        alpha2_p = a2
                        changed = True
                else:
                    finished = True
                case3 = True
            elif (not case4 and
                  (alpha1_n > 0 or (alpha1_p == 0 and delta_E < 0)) and
                  (alpha2_n > 0 or (alpha2_p == 0 and delta_E > 0))):
                # compute L and H wrt alpha1_n, alpha2_n
                L = max(0, -gamma - self.C)
                H = min(self.C, -gamma)
                if L < H:
                    if eta > 0:
                        a2 = max(L, min(alpha2_n + delta_E / eta, H))
                    else:
                        Lobj = L * delta_E
                        Hobj = H * delta_E
                        a2 = L if Lobj > Hobj else H
                        warnings.warn('kernel matrix is not positive definite', PositiveSpectrumWarning)
                    a1 = alpha1_n - (a2 - alpha2_n)
                    # update alpha1_n, alpha2_n if change is larger than some eps
                    if abs(a1 - alpha1_n) > 1e-12 or abs(a2 - alpha2_n) > 1e-12:
                        alpha1_n = a1
                        alpha2_n = a2
                        changed = True
                else:
                    finished = True
                case4 = True
            else:
                finished = True

            delta_E += eta * ((alpha2_p - alpha2_n) - (self.alphas_p[i2] - self.alphas_n[i2]))

        if not changed:
            return False

        # if kernel is liner update weight vector
        # to reflect change in a1 and a2
        if isinstance(self.kernel, LinearKernel):
            self.w -= (((self.alphas_p[i1] - self.alphas_n[i1]) - (alpha1_p - alpha1_n)) * self.X[i1] +
                       ((self.alphas_p[i2] - self.alphas_n[i2]) - (alpha2_p - alpha2_n)) * self.X[i2])

        # update error cache using new alphas
        for i in self.I0:
            if i != i1 and i != i2:
                self.errors[i] += (
                        ((self.alphas_p[i1] - self.alphas_n[i1]) - (alpha1_p - alpha1_n)) * self.K[i1, i] +
                        ((self.alphas_p[i2] - self.alphas_n[i2]) - (alpha2_p - alpha2_n)) * self.K[i2, i])
        # update error cache using new alphas for i1 and i2
        self.errors[i1] += (((self.alphas_p[i1] - self.alphas_n[i1]) - (alpha1_p - alpha1_n)) * self.K[i1, i1] +
                            ((self.alphas_p[i2] - self.alphas_n[i2]) - (alpha2_p - alpha2_n)) * self.K[i1, i2])
        self.errors[i2] += (((self.alphas_p[i1] - self.alphas_n[i1]) - (alpha1_p - alpha1_n)) * self.K[i1, i2] +
                            ((self.alphas_p[i2] - self.alphas_n[i2]) - (alpha2_p - alpha2_n)) * self.K[i2, i2])

        # to prevent precision problems
        if alpha1_p > self.C - 1e-10 * self.C:
            alpha1_p = self.C
        elif alpha1_p <= 1e-10 * self.C:
            alpha1_p = 0

        if alpha1_n > self.C - 1e-10 * self.C:
            alpha1_n = self.C
        elif alpha1_n <= 1e-10 * self.C:
            alpha1_n = 0

        if alpha2_p > self.C - 1e-10 * self.C:
            alpha2_p = self.C
        elif alpha2_p <= 1e-10 * self.C:
            alpha2_p = 0

        if alpha2_n > self.C - 1e-10 * self.C:
            alpha2_n = self.C
        elif alpha2_n <= 1e-10 * self.C:
            alpha2_n = 0

        # update model object with new alphas
        self.alphas_p[i1], self.alphas_p[i2] = alpha1_p, alpha2_p
        self.alphas_n[i1], self.alphas_n[i2] = alpha1_n, alpha2_n

        # update the sets of indices for i1 and i2
        for i in (i1, i2):
            if 0 < self.alphas_p[i] < self.C or 0 < self.alphas_n[i] < self.C:
                self.I0.add(i)
            else:
                self.I0.discard(i)
            if self.alphas_p[i] == 0 and self.alphas_n[i] == 0:
                self.I1.add(i)
            else:
                self.I1.discard(i)
            if self.alphas_p[i] == 0 and self.alphas_n[i] == self.C:
                self.I2.add(i)
            else:
                self.I2.discard(i)
            if self.alphas_p[i] == self.C and self.alphas_n[i] == 0:
                self.I3.add(i)
            else:
                self.I3.discard(i)

        # update thresholds
        self.b_up_idx = -1
        self.b_low_idx = -1
        self.b_up = sys.float_info.max
        self.b_low = -sys.float_info.max

        for i in self.I0:
            if 0 < self.alphas_p[i] < self.C and self.errors[i] - self.epsilon > self.b_low:
                self.b_low = self.errors[i] - self.epsilon
                self.b_low_idx = i
            elif 0 < self.alphas_n[i] < self.C and self.errors[i] + self.epsilon > self.b_low:
                self.b_low = self.errors[i] + self.epsilon
                self.b_low_idx = i

            if 0 < self.alphas_p[i] < self.C and self.errors[i] - self.epsilon < self.b_up:
                self.b_up = self.errors[i] - self.epsilon
                self.b_up_idx = i
            elif 0 < self.alphas_n[i] < self.C and self.errors[i] + self.epsilon < self.b_up:
                self.b_up = self.errors[i] + self.epsilon
                self.b_up_idx = i

        for i in (i1, i2):
            if i not in self.I0:
                if i in self.I2 and self.errors[i] + self.epsilon > self.b_low:
                    self.b_low = self.errors[i] + self.epsilon
                    self.b_low_idx = i
                elif i in self.I1 and self.errors[i] - self.epsilon > self.b_low:
                    self.b_low = self.errors[i] - self.epsilon
                    self.b_low_idx = i

                if i in self.I3 and self.errors[i] - self.epsilon < self.b_up:
                    self.b_up = self.errors[i] - self.epsilon
                    self.b_up_idx = i
                elif i in self.I1 and self.errors[i] + self.epsilon < self.b_up:
                    self.b_up = self.errors[i] + self.epsilon
                    self.b_up_idx = i

        if self.b_low_idx == -1 or self.b_up_idx == -1:
            raise Exception('unexpected status')

        return True

    def _examine_example(self, i2):
        alpha2_p, alpha2_n = self.alphas_p[i2], self.alphas_n[i2]

        if i2 in self.I0:
            E2 = self.errors[i2]
        else:
            E2 = self.y[i2] - (self.alphas_p - self.alphas_n).dot(self.K[i2])
            self.errors[i2] = E2
            # update (b_low, b_low_idx) or (b_up, b_up_idx) using (E2, i2)
            if i2 in self.I1:
                if E2 + self.epsilon < self.b_up:
                    self.b_up = E2 + self.epsilon
                    self.b_up_idx = i2
                elif E2 - self.epsilon > self.b_low:
                    self.b_low = E2 - self.epsilon
                    self.b_low_idx = i2
            elif i2 in self.I2 and E2 + self.epsilon > self.b_low:
                self.b_low = E2 + self.epsilon
                self.b_low_idx = i2
            elif i2 in self.I3 and E2 - self.epsilon < self.b_up:
                self.b_up = E2 - self.epsilon
                self.b_up_idx = i2

        # check optimality using current b_up and b_low and, if violated,
        # find another index i1 to do joint optimization with i2
        i1 = -1
        optimal = True
        if i2 in self.I0:
            if 0 < alpha2_p < self.C:
                if self.b_low - (E2 - self.epsilon) > 2 * self.tol:
                    optimal = False
                    i1 = self.b_low_idx
                    # for i2 in I0 choose the better i1
                    if (E2 - self.epsilon) - self.b_up > self.b_low - (E2 - self.epsilon):
                        i1 = self.b_up_idx
                elif (E2 - self.epsilon) - self.b_up > 2 * self.tol:
                    optimal = False
                    i1 = self.b_up_idx
                    # for i2 in I0 choose the better i1
                    if self.b_low - (E2 - self.epsilon) > (E2 - self.epsilon) - self.b_up:
                        i1 = self.b_low_idx
            elif 0 < alpha2_n < self.C:
                if self.b_low - (E2 + self.epsilon) > 2 * self.tol:
                    optimal = False
                    i1 = self.b_low_idx
                    # for i2 in I0 choose the better i1
                    if (E2 + self.epsilon) - self.b_up > self.b_low - (E2 + self.epsilon):
                        i1 = self.b_up_idx
                elif (E2 + self.epsilon) - self.b_up > 2 * self.tol:
                    optimal = False
                    i1 = self.b_up_idx
                    # for i2 in I0 choose the better i1
                    if self.b_low - (E2 + self.epsilon) > (E2 + self.epsilon) - self.b_up:
                        i1 = self.b_low_idx
        elif i2 in self.I1:
            if self.b_low - (E2 + self.epsilon) > 2 * self.tol:
                optimal = False
                i1 = self.b_low_idx
                # for i2 in I1 choose the better i1
                if (E2 + self.epsilon) - self.b_up > self.b_low - (E2 + self.epsilon):
                    i1 = self.b_up_idx
            elif (E2 - self.epsilon) - self.b_up > 2 * self.tol:
                optimal = False
                i1 = self.b_up_idx
                # for i2 in I1 choose the better i1
                if self.b_low - (E2 - self.epsilon) > (E2 - self.epsilon) - self.b_up:
                    i1 = self.b_low_idx
        elif i2 in self.I2:
            if (E2 + self.epsilon) - self.b_up > 2 * self.tol:
                optimal = False
                i1 = self.b_up_idx
        elif i2 in self.I3:
            if self.b_low - (E2 - self.epsilon) > 2 * self.tol:
                optimal = False
                i1 = self.b_low_idx
        else:
            raise Exception('the index could not be found')

        if optimal:
            return False

        return self._take_step(i1, i2)

    def minimize(self):
        if self.verbose:
            print('iter\t cost')

        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            # loop over all training examples
            if examine_all:
                for i in range(len(self.X)):
                    num_changed += self._examine_example(i)
            else:
                # loop over examples where alphas are not already at their limits
                for i in range(len(self.X)):
                    if 0 < self.alphas_p[i] < self.C or 0 < self.alphas_n[i] < self.C:
                        num_changed += self._examine_example(i)
                        # check if optimality on I0 is attained
                        if self.b_up > self.b_low - 2 * self.tol:
                            num_changed = 0
                            break
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            if self.verbose and not self.iter % self.verbose:
                print('{:4d}\t{: 1.4e}'.format(
                    self.iter, self.quad.function(np.concatenate((self.alphas_p, self.alphas_n)))))

            self.iter += 1

        self.b = (self.b_low + self.b_up) / 2

        if self.verbose:
            print()

        return self
