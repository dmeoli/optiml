from . import StochasticMomentumOptimizer


class StochasticGradientDescent(StochasticMomentumOptimizer):

    def __init__(self,
                 f,
                 x=None,
                 batch_size=None,
                 eps=1e-6,
                 tol=1e-8,
                 epochs=1000,
                 step_size=0.01,
                 momentum_type='none',
                 momentum=0.9,
                 callback=None,
                 callback_args=(),
                 shuffle=True,
                 random_state=None,
                 verbose=False):
        super(StochasticGradientDescent, self).__init__(f=f,
                                                        x=x,
                                                        step_size=step_size,
                                                        momentum_type=momentum_type,
                                                        momentum=momentum,
                                                        batch_size=batch_size,
                                                        eps=eps,
                                                        tol=tol,
                                                        epochs=epochs,
                                                        callback=callback,
                                                        callback_args=callback_args,
                                                        shuffle=shuffle,
                                                        random_state=random_state,
                                                        verbose=verbose)

    def minimize(self):

        self._print_header()

        for batch in self.batches:

            if self.momentum_type == 'nesterov':
                step_m1 = self.step
                jump = next(self.momentum) * step_m1
                self.x += jump

            self.f_x, self.g_x = self.f.function_jacobian(self.x, *batch)

            self._print_info()

            try:
                self.callback(batch)
            except StopIteration:
                break

            if self.is_batch_end():
                self.epoch += 1

            if self.epoch >= self.epochs:
                self.status = 'stopped'
                break

            # compute search direction
            d = -self.g_x

            if self.momentum_type == 'polyak':

                step_m1 = self.step
                self.step = next(self.step_size(*batch)) * d + next(self.momentum) * step_m1
                self.x += self.step

            elif self.momentum_type == 'nesterov':

                correction = next(self.step_size(*batch)) * d
                self.x += correction
                self.step = jump + correction

            elif self.momentum_type == 'none':

                self.step = next(self.step_size(*batch)) * d
                self.x += self.step

            try:
                self.check_lagrangian_dual_optimality()
            except StopIteration:
                break

            self.iter += 1

        self.check_lagrangian_dual_conditions()

        if self.verbose:
            print('\n')

        return self
