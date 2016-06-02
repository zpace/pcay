'''
Define several stellar initial mass functions,
    with some tools for working with them
'''

class IMF(object):
    '''
    stellar initial mass function
    '''

    __version__ = '0.1'

    def __init__(self, imftype='salpeter', ml=0.1, mh=150., mf=1., dm=.005):
        '''
        set up an IMF with some probability distribution, lower mass limit,
            and upper mass limit, that formed some mass

        all masses & luminosities are implicitly in solar units, and times
            are in Gyr

        I've provided several choices of IMF
        '''

        self.imftype = imftype
        self.ml = ml # low mass limit
        self.mh = mh # high mass limit
        self.dm = dm # standard mass differential for computations

    @staticmethod
    def salpeter_pdf_u(m):
        '''straight up power law'''
        return 1./2.28707 * m**-2.35

    @staticmethod
    def miller_scalo_pdf_u(m):
        bdy = 1.
        inds = [0., -2.3]
        branch = np.argmax(np.stack([]))
        return m**-2.35

    @staticmethod
    def kroupa_u(m):
        bdys = [.08, .5]
        inds = [-0.3, -1.3, -2.3]

    @staticmethod
    def chabrier_u(m):

    def mass_at_age(t):
        raise NotImplementedError




