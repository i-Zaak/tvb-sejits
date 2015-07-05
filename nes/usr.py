from sympy import ProductSet, FiniteSet, Range, EmptySet

class USR:
    '''
    Uniform Set of References, aka RT_LMAD (Real Time Linear Memory Access
    Descriptor) both to be found in Rus et al. 2003 and Rus et al. 2008. Simple
    implementation for the needs of TVB. No strides, branching, conditionals
    etc., only union, intersection, complement and subtration; uses sympy
    finite sets as underlying implementation of dimensions of known sizes as a
    quick fix and ignores symbolic dimensions altogether. MD-LMAD
    (Multi-Dimensional Linear Memory Access Descriptor (Paek et al. 2002))
    leaves represented by ProductSet of Range instances (seems to be exact). To
    be rewritten and/or replaced with full RT_LMAD implementation if needed in
    future ;)
    '''

    def __init__(self, subscripts=None):
        '''
        Creates an empty USR or an USR containing single MD-LMAD leaf
        

        subscripts: 
            list of subscripts defining the access to particular dimension, for
            every dimension we use a triplet notation: lower bound, upper
            bound, stride. Each subscript is therefore a tuple of following
            possible formats: n - linear range from 0 to n-1; (low, up) -
            linear range from low to up-1; (low, up,step) same, but with a
            step.
        '''
        if subscripts is None:
            self._symbolic_set = EmptySet()
        else:
            ranges = tuple(map(lambda x: Range(x) if isinstance(x,int) else Range(*x), subscripts))
            self._symbolic_set = ProductSet(*ranges)
    
    def intersect(self, other):
        usr = USR()
        usr._symbolic_set = self._symbolic_set.intersect(other)
        return usr

    def union(self, other):
        usr = USR()
        usr._symbolic_set = self._symbolic_set.union(other)
        return usr

    def is_empty(self):
        return FiniteSet(self._symbolic_set) is EmptySet()

    def subtract(self, other):
        usr = USR()
        usr._symbolic_set = FiniteSet(self._symbolic_set) - FiniteSet(other._symbolic_set)
        return usr

    def __eq__(self, other):
        return FiniteSet(self._symbolic_set) == FiniteSet(other._symbolic_set)
