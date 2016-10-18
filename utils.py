import numpy as np
import pickle as pkl

class ArrayPartitioner(object):
    '''
    partition arrays along an axis, and return an iterator

    Iterator produces all the subarrays of `arrays` along LAST TWO dimensions

    Parameters:
    -----------

    arrays : list
        list of arrays to partition identically

        Each of these arrays must have identical shape in final two dimensions

    lgst_ix : int (default: 0)
        index of `arrays` that corresponds to the array that limits the size
        of individual operations

    lgst_el_shape : tuple
        shape of each intermediate array element that needs to be handled

    memlim : int
        limit of memory for each block of several lgst_el_size to take up

    Example:
    --------

    `lim_array` has shape (47, 47, 24, 24), and `arrays` has elements with
    shape (..., 24, 24) (the ... could be nothing). In this case,
    `lgst_el_shape` = (24, 24). The iterator will return portions of
    the len-(47*47=2209) "flattened" array
    '''

    def __init__(self, arrays, lgst_el_shape, lgst_ix=0, memlim=2147483648):
        self.elshape = lgst_el_shape
        self.imshape = arrays[lgst_ix].shape[-2:]

        def f(a):
            if len(a.shape) == 2:
                return a.flatten()
            else:
                return np.moveaxis(a.reshape((self.elshape +
                                              (np.prod(self.imshape), ))),
                                   [0, 1, 2], [1, 2, 0])

        self.arrays = [f(a) for a in arrays]

        # calculate the max number of elements of size lgst_el_size
        # size in memory of largest intermediate element
        lgst_el_size = np.empty(lgst_el_shape).nbytes
        # how many intermediate subarrays are possible to
        # store at once given memlim
        self.M = memlim // lgst_el_size

        # the basic idea is that we get N partitions such that each partition
        # has at most size M
        self.N = np.prod(self.imshape) // self.M

        self.ct = 0  # sentinel value

    def __iter__(self):
        return self

    def __next__(self):
        if self.ct > self.N:
            raise StopIteration
        elif self.ct == self.N:
            r = (a[..., (np.prod(self.imshape) - self.M * self.N):]
                 for a in self.arrays)
        else:
            r = (a[..., (self.ct * self.M):((self.ct + 1) * self.M)]
                 for a in self.arrays)

        self.ct += 1

        return r

def pickle_loader(fname):
    with open(fname, 'rb') as f:
        p = pkl.load(f)

    return p
