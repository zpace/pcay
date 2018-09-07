import numpy as np
eps = np.finfo(float).eps

class ParamInterpMap(object):
    '''
    interpolator for a parameter based on weights on samples
    '''
    def __init__(self, v, w):

        # sort values of parameter
        order = np.argsort(v, axis=0)

        # apply sort to values and then weights
        self.v_o = v[order]
        w_sum = w.sum(axis=0, keepdims=True)
        w = w + eps * np.isclose(w_sum, 0, atol=eps)

        w_o = w[order] + eps

        self.cumpctl = 100. * (np.cumsum(w_o, axis=0) - w_o / 2) / \
            np.sum(w_o, axis=0, keepdims=True)

    def find_pctl_pos(self, pctl):
        '''
        find the index where pctl would lay, which corresponds to rhs index
        '''
        i_rhs = np.apply_along_axis(
            lambda arr: arr.searchsorted(pctl, side='right'),
            axis=0, arr=self.cumpctl).clip(0, self.cumpctl.shape[0] - 1)

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

class LargeNDInterpolator(object):
    '''
    interpolates an array along a single axis
    '''
    def __init__(self, arr, axis_coords, interp_axis=0):
        '''
        - arr: array to be interpolated
        - axis_coords: coordinate array, with same shape as arr, that increments
              along interp_axis
        - interp_axis: axis along which interpolation is carried out
        '''

        self.arr = arr
        self.shape = arr.shape
        self.interp_axis = interp_axis
        self.axis_coords = axis_coords

    def find_coord_pos(self, coord):
        '''
        find the position of a **single** coordinate location along interp_axis
        '''
        ix_rhs = np.apply_along_axis(
            lambda arr: arr.searchsorted(coord, side='right'),
            axis=self.interp_axis, arr=self.axis_coords)
        return ix_rhs

    def val_at_coord(self, coord):
        '''
        find the value at a **single** coordinate location along interp_axis
        '''

        # indices of rhs and lhs bounds
        i1 = self.find_coord_pos(coord)
        # if i1 is too large, reduce it to maximum allowable
        outofbounds = (i1 >= self.arr.shape[self.interp_axis])
        i1[outofbounds] = self.arr.shape[self.interp_axis] - 1
        i0 = i1 - 1

        # values at lhs and rhs bounds
        II, JJ = np.meshgrid(*map(np.arange, ))
        v1 = self.arr.take(indices=i1, axis=self.interp_axis, mode='clip')
        v0 = self.arr.take(indices=i0, axis=self.interp_axis, mode='clip')

        c1 = self.axis_coords.take(indices=i1, axis=self.interp_axis, mode='clip')
        c0 = self.axis_coords.take(indices=i0, axis=self.interp_axis, mode='clip')

        val = v0 + ((coord - c0) / (c1 - c0)) * (v1 - v0)
        val[c1 == c0] = v0[c1 == c0]

        return val

    def __call__(self, coords, signature=None):
        res = np.vectorize(self.val_at_coord, signature=signature)(coords)
        return res

class ParamInterpMap2(LargeNDInterpolator):
    '''
    interpolator for a parameter based on weights on samples
    '''
    def __init__(self, v, w, axis=0):
        # sort values of parameter
        order = np.argsort(v)

        # apply sort to values and then weights
        v_o = v.take(order)
        w_o = w.take(order, axis=axis) + np.finfo(w.dtype).eps
        # move interpolation axis to final position
        w_o = np.moveaxis(w_o, axis, -1)
        self.v_o_broadcast, _ = np.broadcast_arrays(v_o, w_o)
        print(self.v_o_broadcast.shape)

        self.cumpctl = 100. * (np.cumsum(w_o, axis=axis) - w_o / 2) / \
            np.sum(w_o, axis=axis, keepdims=True)

        super().__init__(arr=self.v_o_broadcast, axis_coords=self.cumpctl, interp_axis=-1)

    def __call__(self, pctls=None, qtls=None):
        # parse quantile or percentile input
        if (pctls is None) and (qtls is None):
            raise ValueError('Must specify percentiles or quantiles')
        elif (pctls is not None) and (qtls is not None):
            raise ValueError('Both percentiles and quantiles specified')
        elif (qtls is not None):
            pctls = 100. * np.array(qtls)

        signature = '()->(n,m)'

        res = super().__call__(pctls, signature=signature)

        return res


class SFHInterpAtTime(LargeNDInterpolator):
    '''
    interpolator for figuring out percentiles of many weighted SFHs at many times
    '''
    def __init__(self, sfrs, w, axis=1):
        # sort values of sfr at each time bin
        order = np.argsort(sfrs, axis=axis)
        self.sfrs_o = sfrs.take(order, axis=axis)
        # apply same sorting to weights
        w_o = w.take(order, axis=axis) + np.finfo(w.dtype).eps

        self.cumpctl = 100. * (np.cumsum(w_o, axis=axis) - w_o / 2) / \
            np.sum(w_o, axis=axis, keepdims=True)

        super().__init__(arr=self.sfrs_o, axis_coords=self.cumpctl, interp_axis=axis)

    def __call__(self, pctls=None, qtls=None):
        # parse quantile or percentile input
        if (pctls is None) and (qtls is None):
            raise ValueError('Must specify percentiles or quantiles')
        elif (pctls is not None) and (qtls is not None):
            raise ValueError('Both percentiles and quantiles specified')
        elif (qtls is not None):
            pctls = 100. * qtls

        signature = '()->(n,)'

        res = super().__call__(pctls, signature=signature)

        return res
