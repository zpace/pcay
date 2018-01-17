import numpy as np

class ParamInterpMap(object):
    '''
    interpolator for a parameter based on weights on samples
    '''
    def __init__(self, v, w):

        # sort values of parameter
        order = np.argsort(v, axis=0)

        # apply sort to values and then weights
        self.v_o = v[order]
        w_o = w[order]

        self.cumpctl = 100. * (np.cumsum(w_o, axis=0) - w_o / 2) / \
            np.sum(w_o, axis=0, keepdims=True)

    def find_pctl_pos(self, pctl):
        '''
        find the index where pctl would lay, which corresponds to rhs index
        '''
        i_rhs = np.apply_along_axis(
            lambda arr: arr.searchsorted(pctl), axis=0, arr=self.cumpctl)

        return i_rhs

    def val_at_pctl(self, pctl):
        '''
        find the value at the given percentile through linear interpolation
        '''

        # indices of lhs and rhs bounds
        i1 = self.find_pctl_pos(pctl)
        i0 = i1 - 1

        # values at lhs and rhs bounds
        v1 = self.v_o[i1]
        v0 = self.v_o[i0]

        # percentile at lhs and rhs bounds
        II, JJ = np.meshgrid(*map(np.arange, self.cumpctl.shape[1:]),
                             indexing='ij')
        p1 = self.cumpctl[i1, II, JJ]
        p0 = self.cumpctl[i0, II, JJ]

        # value at specified percentile is constructed from percentiles
        # and values at those percentiles
        val = v0 + ((pctl - p0) / (p1 - p0)) * (v1 - v0)

        return val

    def __call__(self, pctls=None, qtls=None):
        # parse quantile or percentile input
        if (pctls is None) and (qtls is None):
            raise ValueError('Must specify percentiles or quantiles')
        elif (pctls is not None) and (qtls is not None):
            raise ValueError('Both percentiles and quantiles specified')
        elif (qtls is not None):
            pctls = 100. * qtls

        res = np.vectorize(self.val_at_pctl, signature='()->(n,m)')(pctls)
        return res

