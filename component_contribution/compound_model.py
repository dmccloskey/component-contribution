from .compound_cacher import CompoundCacher

class compound_model(object):
    """description of class"""
    
    def __del__(self):
        self.ccache.dump()
    
    def __init__(self, cids, dG0_f):
        self.cids = cids
        self.dG0_f = dG0_f
        self.ccache = CompoundCacher()

    def get_transformed_dG0_f(self, pH, I, T):
        """
            returns the estimated dG0_prime for the compounds
            and the standard deviation of each estimate (i.e. a measure for the uncertainty)
            for the compounds.
        """

        dG0_prime_f_tmp = self._get_transform_ddG0_f(pH=pH, I=I, T=T)
        dG0_prime_f = []
        for i in range(len(self.dG0_f)): 
            dG0_prime_f.append(self.dG0_f[i] + dG0_prime_f_tmp[i])
               
        return dG0_prime_f

    def _get_transform_ddG0_f(self, pH, I, T):
        """
        needed in order to calculate the transformed Gibbs energies of the 
        model compounds.
        
        Returns:
            an array (whose length is self.S.shape[1]) with the differences
            between DrG0_prime and DrG0. Therefore, one must add this array
            to the chemical Gibbs energies of reaction (DrG0) to get the 
            transformed values
        """

        dG0_prime_f = []
        for i, cid in enumerate(self.cids):
            comp = self.ccache.get_kegg_compound_f(cid)
            dG0_prime_f.append(comp.transform(len(comp.zs)-1, pH, I, T))

        return dG0_prime_f # return only the transformed compound dG0


