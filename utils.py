import numpy as np

class ArrayPartitioner(object):
    '''
    partition arrays along an axis, and return an iterator

    Iterator produces all the elements of

    Note: this operation always preserves overall number of axes

    Parameters:
    -----------

    arrays : list
        list of arrays to partition

    lgst_el_shape : tuple
        largest element shape of intermediate array that needs to be handled

    axis : int or tuple
        axis or axes along which to partition the input arrays

    memlim : int
        limit of memory for each block of several lgst_el_size to take up
    '''

    def __init__(self, arrays, lgst_el_size, axis=(-2, -1), memlim=2147483648):

        # calculate the max number of elements of size lgst_el_size
        # size in memory of largest intermediate element
        lgst_el_size = np.empty(lgst_el_shape).nbytes
        # how many are possible given memlim
        self.M = memlim // lgst_el_size

        # figure out how many steps are needed
        a0 = arrays[0]
        a0_el = a0[list(axis)]
        self.N =

    def _maskrule(self):
        '''
        how to mask
        '''

    def __iter__(self):
        return self

    def __next__(self):
        if self.ct == self.N:
            raise StopIteration

        self.ct += 1

        return (a[self._maskrule()] for a in arrays)

