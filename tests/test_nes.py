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
        py_ast = ast.parse("x = 2 + b * c")
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
    def assign_test(self):
        py_ast = ast.parse("x = a\nb =  x + c")
        dfdag = nes.ast_to_dfdag(py_ast)

        
        a = Value()
        b = Value()
        c = Value()
        add1 = Apply(BinOp([None],None),[a,c], b) 

        dfdag_ex = DFDAG([add1], [a,b,c])

        self.assertTrue( graphs_isomorphic(dfdag, dfdag_ex) )


    def slice_test(self):
        py_ast = ast.parse("x[0]")
        dfdag = nes.ast_to_dfdag(py_ast, variable_shapes = {'x': ('svar','nodes','modes')})

        exp_shape = ('nodes','modes')
        self.assertTrue(dfdag.values[0].type.shape ==exp_shape or 
                dfdag.values[1].type.shape ==exp_shape)

    def slice_multidim_test(self):
        py_ast = ast.parse("x[0,:,2]")
        dfdag = nes.ast_to_dfdag(py_ast, variable_shapes = {'x': ('svar','nodes','modes')})

        exp_shape = ('nodes',)
        self.assertTrue(dfdag.values[0].type.shape ==exp_shape or 
                dfdag.values[1].type.shape ==exp_shape)

    def type_propagation_test(self):
        py_ast = ast.parse("x = a  + b")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'a': ('svar','nodes','modes'),
                    'b': ('svar','nodes','modes'),
                    })
        
        exp_shape = ('svar','nodes','modes')
        self.assertTrue(dfdag.values[0].type.shape ==exp_shape)
        self.assertTrue(dfdag.values[1].type.shape ==exp_shape)
        self.assertTrue(dfdag.values[2].type.shape ==exp_shape)

    def type_broadcasts_tests(self):
        py_ast = ast.parse("x = a  + b")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'a': ('svar','nodes','modes'),
                    'b': 'scalar',
                    })
        exp_shape = ('svar','nodes','modes')
        self.assertTrue(dfdag.applies[0].output.type.shape == exp_shape)
    def type_scalar_tests(self):
        py_ast = ast.parse("x = a  + b")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'a': 'scalar',
                    'b': 'scalar',
                    })
        self.assertTrue(isinstance(dfdag.applies[0].output.type, ScalarType))

    def type_slicing_test(self):
        py_ast = ast.parse("a = c[0]\nb = c[1]\nx = a + b")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'c': ('cvar','nodes','modes'),
                    'x': ('nodes','modes')
                    })
        
        self.assertTrue(dfdag.applies[0].output.type.shape == ('nodes', 'modes'))
        self.assertTrue(dfdag.applies[0].output.type.slice == ('nodes', 'modes'))
        self.assertTrue(dfdag.applies[0].output.type.data.shape == ('nodes', 'modes'))

        self.assertTrue(dfdag.applies[0].inputs[0].type.shape == ('nodes', 'modes'))
        self.assertTrue(dfdag.applies[0].inputs[0].type.slice == (0, 'nodes', 'modes'))
        self.assertTrue(dfdag.applies[0].inputs[0].type.data.shape == ('cvar', 'nodes', 'modes'))
        self.assertTrue(dfdag.applies[0].inputs[0].type.data == dfdag.applies[0].inputs[0].type.data)


        
    def sliced_assign_test(self):
        py_ast = ast.parse("a[0] = x + c\na[1] = x - c\nb=a+1")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'a': ('cvar','nodes','modes'),
                    'c': 'scalar',
                    'x': ('nodes','modes')
                    })
        import ipdb; ipdb.set_trace()


       
