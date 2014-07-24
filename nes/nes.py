"""
Specializers for neural ensamble models. To be used together with TVB.
"""

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import numpy as np

class CModelDfunFunction(ConcreteSpecializedFunction):
    def finalize(self, program, tree, entry_name):
        self._c_function = self._compile(program, tree, entry_name)
    def __call__(self, state_variables, coupling, local_coupling):
        self._c_function(state_variables, coupling, local_coupling)


class CModelDfun(LazySpecializedFunction):
    def transform(self, tree, program_config):
        #traverse the python AST, replace numpy slices with loops

        fn = CModelDfunFunction()
        return fn.finalize(program, Project([tree]),"dfun")

