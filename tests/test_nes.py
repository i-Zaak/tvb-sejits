import unittest

from nes.dfdag import *
import ast
from nes import nes

import networkx.algorithms.isomorphism as iso

import networkx as nx



def graphs_isomorphic(dfdag1, dfdag2):
    """
    Graphs are isomorphic and the nodes are equal. We should test the tests...
    """
    g1 = dfdag1.nx_representation()
    g2 = dfdag2.nx_representation()
    # TODO add a generic node match helper...
    return iso.is_isomorphic(g1,g2) 




class AstParsingTest(unittest.TestCase):
    def simple_test(self):
        py_ast = ast.parse("x = a + b * c")
        dfdag = nes.ast_to_dfdag(py_ast)

        
        a = Value()
        b = Value()
        c = Value()

        t1 = Value()
        mult = Apply(BinOp([None],None),[b,c], t1) #no types for now
        x = Value()
        plus = Apply(BinOp([None],None),[a, t1], x) #no types for now

        dfdag_ex = DFDAG([mult,plus], [a,b,c,t1,x])

        self.assertTrue( graphs_isomorphic(dfdag, dfdag_ex) )

    def multiline_test(self):
        py_ast = ast.parse("x = a + b\ny = x * b")
        dfdag = nes.ast_to_dfdag(py_ast)
        
        a = Value()
        b = Value()

        x = Value()
        y = Value()
        add1 = Apply(BinOp([None],None),[a,b], x) 
        add2 = Apply(BinOp([None],None),[x,b], y) 

        dfdag_ex = DFDAG([add1, add2], [a,b,x,y])

        self.assertTrue( graphs_isomorphic(dfdag, dfdag_ex) )

    def slice_test(self):
        py_ast = ast.parse("x[0]")
        dfdag = nes.ast_to_dfdag(py_ast, variable_shapes = {'x': ['svar','nodes','modes']})

        exp_shape = [0,'nodes','modes']
        self.assertTrue(dfdag.values[0].type.shape ==exp_shape)

    def slice_multidim_test(self):
        py_ast = ast.parse("x[0,:,2]")
        dfdag = nes.ast_to_dfdag(py_ast, variable_shapes = {'x': ['svar','nodes','modes']})

        exp_shape = [0,'nodes',2]
