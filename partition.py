import numpy as np
from functools import reduce
from itertools import product

def factors(n):
    return set(reduce(
        list.__add__, ([i, n // i] for i in range(1, int(n**0.5) + 1)
                        if n % i == 0)))

def factor_pairs(n):
    f = factors(n)
    f_array = np.array(list(f))
    f_cofactor = n // f_array
    f_pairs = np.column_stack([f_array, f_cofactor])
    return tuple((nchunk, chunklen)
                 for nchunk, chunklen in zip(f_array, f_cofactor))

def nsub_subshape(chunkspec1, chunkspec2):
    '''
    for a pair of chunkspec tuples (# of chunks along axis, # of elements per chunk),
        return the shape of the "chunk array" & the shape of an individual chunk
    '''

    return (chunkspec1[0], chunkspec2[0]), (chunkspec1[1], chunkspec2[1])

def find_largest_good_chunkshape(chunkspecs_I, chunkspecs_J, nl):
    lgst_chunksize = (0, 0)
    nchunksperaxis = (0, 0)
    for chunkspec_i, chunkspec_j in product(chunkspecs_I, chunkspecs_J):
        nsub, subshape = nsub_subshape(chunkspec_i, chunkspec_j)
        if try_arraysize(shape=(nl, nl) + subshape, dtype=np.float):
            if np.product(subshape) > np.product(lgst_chunksize):
                lgst_chunksize = subshape
                nchunksperaxis = nsub

    return lgst_chunksize, nchunksperaxis

def extract_fixedlength_subarray(large_array, i0, n):
    '''
    extract a cube into a smaller cube
    '''
    mapshape = i0.shape
    II, JJ = np.meshgrid(*mapshape)
    out = large_array[i0[:, None, None] + np.arange(n), II, JJ]
    return out

def extract_sq_from_sq(large_array, i0, n):
    '''
    extract index interval [i0, i0 + n] from large array into another array
    '''
    mapshape = i0.shape
    II, JJ = np.meshgrid(n, n)
    out = large_array[i0[..., None, None] + II, i0[..., None, None] + JJ]
    return out

def try_arraysize(shape, dtype):
    try:
        test = np.empty(shape, dtype)
    except MemoryError:
        return False
    else:
        return True

def blockgen(array, bpa):
    '''
    Creates a generator that yields multidimensional blocks from the given
    array(_like); bpa is an array_like consisting of the number of blocks per axis
    (minimum of 1, must be a divisor of the corresponding axis size of array). As
    the blocks are selected using normal numpy slicing, they will be views rather
    than copies; this is good for very large multidimensional arrays that are being
    blocked, and for very large blocks, but it also means that the result must be
    copied if it is to be modified (unless modifying the original data as well is
    intended)
    '''

    bpa = np.asarray(bpa) # in case bpa wasn't already an ndarray

    # parameter checking
    if array.ndim != bpa.size:         # bpa doesn't match array dimensionality
        raise ValueError("Size of bpa must be equal to the array dimensionality.")
    if (bpa.dtype != np.int            # bpa must be all integers
        or (bpa < 1).any()             # all values in bpa must be >= 1
        or (array.shape % bpa).any()): # % != 0 means not evenly divisible
        raise ValueError("bpa ({0}) must consist of nonzero positive integers "
                         "that evenly divide the corresponding array axis "
                         "size".format(bpa))


    # generate block edge indices
    rgen = (np.r_[:array.shape[i]+1:array.shape[i]//blk_n]
            for i, blk_n in enumerate(bpa))

    # build slice sequences for each axis (unfortunately broadcasting
    # can't be used to make the items easy to operate over
    c = [[np.s_[i:j] for i, j in zip(r[:-1], r[1:])] for r in rgen]

    # Now to get the blocks; this is slightly less efficient than it could be
    # because numpy doesn't like jagged arrays and I didn't feel like writing
    # a ufunc for it.
    for idxs in np.ndindex(*bpa):
        blockbounds = tuple(c[j][idxs[j]] for j in range(bpa.size))

        yield blockbounds

class CovCalcPartitioner(object):
    def __init__(self, kspec_obs, a_map, i0_map, E, ivar_scaled):
        self.kspec_obs = kspec_obs
        self.a_map = a_map
        self.i0_map = i0_map
        self.mapshape = a_map.shape
        self.nl, self.q = E.shape
        self.E = E
        self.ivar_scaled = ivar_scaled

        self.lgst_chunksize, self.nchunksperaxis = self._how_to_chunkify_maps()
        self.blockgen = blockgen(ivar_scaled, (1, ) + self.nchunksperaxis)

    def _how_to_chunkify_maps(self):
        chunkspecsI, chunkspecsJ = tuple(map(factor_pairs, self.mapshape))
        lgst_chunksize, nchunksperaxis = find_largest_good_chunkshape(
            chunkspecsI, chunkspecsJ, nl)
        return lgst_chunksize, nchunksperaxis

    def calc(self, i0, a, var):
        '''
        vector calculation of PC covariance matrix
        '''
        # retrieve appropriate obs cov for each spaxel
        kspec_obs_sub = extract_sq_from_sq(self.kspec_obs, i0, self.nl)
        # replace main diag of each spaxel's k_obs with its variance array
        np.einsum('abii->abi', kspec)[:] = var

        K_PC_sub = self.E @ kspec_obs_sub @ self.E.T
        return K_PC_sub

    def calc_allchunks(self, K_PC=None):
        '''
        use the blockgen generator to build the full K_PC
        '''
        if K_PC is None:
            K_PC = 100. * np.ones((q, q) + self.mapshape)

        for block in blockbounds:
            cubeblk = next(self.blockgen)
            lamblk, Iblk, Jblk = blk

            mapblk = (Iblk, Jblk)

            i0_ = self.i0_map[mapblk]
            a_ = self.a_map[mapblk]
            var_ = (1. / self.ivar_scaled[cubeblk]).clip(min=1.0e-4, max=1.0e4)

            K_PC[(slice(None, None), slice(None, None)) + mapblk] = self.calc(
                i0=i0_, a=a_, var=var_)

        return K_PC

