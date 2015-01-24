import unittest

from nes.dfdag import *
import ast
from nes import nes

import networkx.algorithms.isomorphism as iso

import networkx as nx



def graphs_equal(dfdag1, dfdag2):
    """
    Graphs are isomorphic and the nodes are equal. We should test the tests...
    """
    g1 = dfdag1.nx_representation()
    for n in g1.nodes():
        g1.node[n]['dfnode'] = n
    g2 = dfdag2.nx_representation()
    for n in g1.nodes():
        g1.node[n]['dfnode'] = n
    node_match = iso.categorical_node_match('dfnode', None)
    
    return iso.is_isomorphic(onx,snx, node_match)




class AstParsingTest(unittest.TestCase):
    def simple_parse_test(self):
        py_ast = ast.parse("x = a + b * c")
        dfdag = nes.ast_to_dfdag(py_ast)

        
        a = Value(ScalarType())
        b = Value(ScalarType())
        c = Value(ScalarType())

        t1 = Value(ScalarType())
        mult = Apply(ScalarBinOp([ScalarType()], [ScalarType()] ),[b,c], t1)
        x = Value(ScalarType())
        plus = Apply(ScalarBinOp([ScalarType()], [ScalarType()] ),[a, t1], x)

        dfdag_ex = DFDAG([mult,plus], [a,b,c,t1,x])


        import ipdb; ipdb.set_trace()
        self.AssertTrue( graphs_equal(dfdag, dfdag_ex) )
