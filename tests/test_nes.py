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
        mult = Apply(BinOp(None,[None],None),[b,c], t1) #no types for now
        x = Value()
        plus = Apply(BinOp(None,[None],None),[a, t1], x) #no types for now

        dfdag_ex = DFDAG([mult,plus], [a,b,c,t1,x])

        self.assertTrue( graphs_isomorphic(dfdag, dfdag_ex) )
    

    def multiline_test(self):
        py_ast = ast.parse("x = a + b\ny = x * b")
        dfdag = nes.ast_to_dfdag(py_ast)
        
        a = Value()
        b = Value()

        x = Value()
        y = Value()
        add1 = Apply(BinOp(None,[None],None),[a,b], x) 
        add2 = Apply(BinOp(None,[None],None),[x,b], y) 

        dfdag_ex = DFDAG([add1, add2], [a,b,x,y])

        self.assertTrue( graphs_isomorphic(dfdag, dfdag_ex) )
    def assign_test(self):
        py_ast = ast.parse("x = a\nb =  x + c")
        dfdag = nes.ast_to_dfdag(py_ast)

        
        a = Value()
        b = Value()
        c = Value()
        add1 = Apply(BinOp(None, [None],None),[a,c], b) 

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
        self.assertTrue(isinstance(dfdag.applies[0].routine.operator, ast.Add))
        self.assertTrue(dfdag.applies[0].routine.output_type.shape == exp_shape)
        self.assertTrue(dfdag.applies[0].routine.input_types[0].shape == exp_shape)
        self.assertTrue(dfdag.applies[0].routine.input_types[1].shape == exp_shape)

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

    def return_test(self):
        py_ast = ast.parse("return x + 1")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'x': ('svar','nodes','modes')
                    })
        self.assertTrue(len(dfdag.results) == 1)
        self.assertTrue(dfdag.results[0].type.shape == ('svar','nodes','modes'))
        
        
    def sliced_assign_test(self):
        py_ast = ast.parse("a[0] = x + c\na[1] = x - c\nreturn a+1")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'a': ('cvar','nodes','modes'),
                    'c': 'scalar',
                    'x': ('nodes','modes')
                    })
        self.assertTrue(len(dfdag.results) == 1)
        self.assertTrue(dfdag.results[0].type.shape == ('cvar','nodes','modes'))
        # how to test the synchronization properly?? 
        # Maybe rolling upwards from return statement?

class VisitorTest(unittest.TestCase):
    def walker_test(self):
        class TestWalker(nes.DFDAGVisitor):
            def __init__(self):
                self.walk = []
                super(TestWalker,self).__init__()
            def visit_Apply(self, node):
                self.walk.append(node.routine)
                self.generic_visit(node) # don't forget these!
            def visit_Value(self,node):
                self.walk.append(node.type)
                self.generic_visit(node) # don't forget these!

        tw = TestWalker()

        a = Value(type=ScalarType())
        b = Value(type=ScalarType())
        c = Value(type=ScalarType())

        plus = Apply(
                BinOp(
                    ast.Add(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                    ),
                [a, b], 
                c)

        dfdag = DFDAG([plus], [a,b,c])

        tw.visit(c)

        self.assertTrue(tw.walk[0] == c.type)
        self.assertTrue(tw.walk[1] == plus.routine)
        self.assertTrue(tw.walk[2] == a.type)
        self.assertTrue(tw.walk[3] == b.type)





class CodeGenTest(unittest.TestCase):
    def value_collector_test(self):
        self.assertTrue(False) # write me

    def bfs_visitor_test(self):
        """
             d-+  f-+  +---------+
        a-a1-c-a2-e-a3-h  g-a4-i-a5-j
        b-+  +--------------+
        """
        a = Value(type=ScalarType())
        b = Value(type=ScalarType())
        c = Value(type=ScalarType())
        d = Value(type=ScalarType())
        e = Value(type=ScalarType())
        f = Value(type=ScalarType())
        g = Value(type=ScalarType())
        h = Value(type=ScalarType())
        i = Value(type=ScalarType())
        j = Value(type=ScalarType())

        a1 = Apply(
                BinOp(ast.Add(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                ),
                [a,b], 
                c)
        a2 = Apply(
                BinOp(ast.Add(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                ),
                [c,d], 
                e)
        a3 = Apply(
                BinOp(ast.Add(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                ),
                [e,f], 
                h)
        a4 = Apply(
                BinOp(ast.Add(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                ),
                [g,c], 
                i)
        a5 = Apply(
                BinOp(ast.Add(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                ),
                [i,h], 
                j)
        dfdag = DFDAG([a1, a2, a3, a4, a5], [a,b,c,d,e,f,g,h,i,j])
        lin = nes.Linearizator(dfdag.applies)
        lin.visit(j)
        self.assertTrue(lin.ordering[0] == a5)
        self.assertTrue(lin.ordering.index(a2) > lin.ordering.index(a3))
        self.assertTrue(lin.ordering.index(a1) > lin.ordering.index(a4))
        self.assertTrue(lin.ordering.index(a1) > lin.ordering.index(a2))
        self.assertTrue(lin.ordering.index(a1) > lin.ordering.index(a3))


    def simple_ctree_test(self):
        a = Value(type=ScalarType())
        b = Value(type=ScalarType())
        c = Value(type=ScalarType())

        t1 = Value()
        mult = Apply(
                BinOp(ast.Mult(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                ),
                [b,c], 
                t1)
        x = Value()
        plus = Apply(
                BinOp(
                    ast.Add(),
                    [ScalarType(), ScalarType()],
                    ScalarType()
                    ),
                [a, t1], 
                x)

        dfdag = DFDAG([mult,plus], [a,b,c,t1,x])
        nes.dfdag_to_ctree(dfdag, None)


