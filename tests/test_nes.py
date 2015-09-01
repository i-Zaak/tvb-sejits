import unittest

from nes.dfdag import *
import ast
from nes import nes

import networkx.algorithms.isomorphism as iso

import networkx as nx
from ctree.jit import JitModule
from ctree import CONFIG
from ctree.c.nodes import CFile, FunctionDecl, Return, SymbolRef
from ctree.cpp.nodes import CppInclude
import ctypes
import numpy as np


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
        df_dag = nes.ast_to_dfdag(py_ast, variable_shapes = {'x': (5,'nodes','modes')})

        exp_shape = ('nodes','modes')
        self.assertTrue(df_dag.values[0].type.shape ==exp_shape or 
                df_dag.values[1].type.shape ==exp_shape)

    def slice_multidim_test(self):
        py_ast = ast.parse("x[0,:,2]")
        df_dag = nes.ast_to_dfdag(py_ast, variable_shapes = {'x': (4,'nodes',3)})

        exp_shape = ('nodes',)
        self.assertTrue(df_dag.values[0].type.shape ==exp_shape or 
                df_dag.values[1].type.shape ==exp_shape)

    def multidim_known_dims_test(self):
        py_ast = ast.parse("x = a +b")
        df_dag = nes.ast_to_dfdag(py_ast, variable_shapes = {
            'a': (4,'nodes',3),
            'b': (4,'nodes',3)
            })
        exp_shape = (4,'nodes',3)
        self.assertTrue(df_dag.applies[0].output.type.data.shape == exp_shape)
        self.assertTrue(df_dag.applies[0].output.type.shape == exp_shape)



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
        df_dag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'c': (3,'nodes','modes'),
                    'x': ('nodes','modes')
                    })
        
        self.assertTrue(df_dag.applies[1].output.type.shape == ('nodes', 'modes'))
        self.assertTrue(df_dag.applies[1].output.type.slice == (':', ':'))
        self.assertTrue(df_dag.applies[1].output.type.data.shape == ('nodes', 'modes'))

        self.assertTrue(df_dag.applies[1].inputs[0].type.shape == ('nodes', 'modes'))
        self.assertTrue(df_dag.applies[1].inputs[0].type.slice == (0, ':', ':'))
        self.assertTrue(df_dag.applies[1].inputs[0].type.data.shape == (3, 'nodes', 'modes'))
        self.assertTrue(df_dag.applies[1].inputs[0].type.data == df_dag.applies[1].inputs[1].type.data)

    def repeated_slice_test(self):
        py_ast = ast.parse("a = x[0]\nb=a[:,1]")
        df_dag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'x': (3,'nodes',5),
                    })
        shapes = set()
        for val in df_dag.values:
            shapes.add(val.type.shape)
        self.assertTrue((3,'nodes',5) in shapes)
        self.assertTrue(('nodes',5) in shapes)
        self.assertTrue(('nodes',) in shapes)


    def return_test(self):
        py_ast = ast.parse("return x + 1")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'x': ('svar','nodes','modes')
                    })
        self.assertFalse(dfdag.result is None)
        self.assertTrue(isinstance(dfdag.result,Value))
        self.assertTrue(dfdag.result.type.shape == ('svar','nodes','modes'))
        
        
    def sliced_assign_test(self):
        py_ast = ast.parse("a[0] = x + c\na[1] = x - c\nreturn a+1")
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'a': (7,'nodes','modes'),
                    'c': 'scalar',
                    'x': ('nodes','modes')
                    })
        self.assertFalse(dfdag.result is None)
        self.assertTrue(dfdag.result.type.shape == (7,'nodes','modes'))
        # how to test the synchronization properly?? 
        # Maybe rolling upwards from return statement?

    def partial_kill_test(self):
        py_ast = ast.parse("c = a + 1\na[0,:] = 1\na[1,:] = 2\nb = a + c")
        df_dag = nes.ast_to_dfdag(
                py_ast, 
                variable_shapes = {
                    'a': (2,'nodes','modes'),
                    })

        # Should test for correct synchronization structure
        self.assertTrue(False, "TODO: write me!")



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





class CtreeBuilderTest(unittest.TestCase):

    def setUp(self):
        from ctree.c.nodes import SymbolRef
        SymbolRef._next_id = 0

    def simple_scalar_test(self):
        py_ast = ast.parse("x = 2 + b * c")
        dfdag = nes.ast_to_dfdag(py_ast, {'b':'scalar','c':'scalar'})
        c_ast = nes.dfdag_to_ctree(dfdag).c_ast
        self.assertTrue(c_ast.codegen() == '// <file: generated.c>\ndouble s_2 = s_1 * s_0;\ndouble s_3 = 2 + s_2;\n')

    def simple_array_test(self):
        py_ast = ast.parse("x = 0.5 + b * c\nreturn x")
        dfdag = nes.ast_to_dfdag(py_ast, {'b':(2,'nodes','modes'),'c':(2,'nodes','modes')})
        ct_builder = nes.dfdag_to_ctree(dfdag)
        ret_sym = ct_builder.return_symbol
        ret_sym.type = np.ctypeslib.ndpointer(dtype=np.float64)()
        c_ast = CFile(body=[
                CppInclude("stdlib.h"),
                FunctionDecl(
                None, "test_fun", 
                params = [
                    SymbolRef("b", np.ctypeslib.ndpointer(dtype=np.float64)()),
                    SymbolRef("c", np.ctypeslib.ndpointer(dtype=np.float64)()),
                    SymbolRef("nodes", ctypes.c_int()),
                    SymbolRef("modes", ctypes.c_int()),
                    ret_sym
                    ],
                defn = ct_builder.c_ast
                )
            ])
        mod = JitModule()
        submod = CFile("test_fun", [c_ast], path=CONFIG.get('jit','COMPILE_PATH'))._compile(c_ast.codegen())
        mod._link_in(submod)
        entry = c_ast.find(FunctionDecl, name="test_fun")             
        c_test_fun = mod.get_callable(entry.name, entry.get_type())     
        nodes = 19
        modes = 5
        a = np.random.rand(2, nodes, modes)
        b = np.random.rand(2, nodes, modes)
        
        res = np.zeros((2,nodes,modes))
        c_test_fun(a,b,nodes,modes,res)
        self.assertTrue(np.allclose( res, 0.5+a*b)) 

    def modified_wilson_cowan_test(self):
        """
        derivative[0] = (tau * (xi - e_i * xi ** 3 / 3.0 - eta) +
                K11 * ((xi/ Aik) - xi) -
                K12 * ((alpha/ Bik) - xi) +
                tau * (IE_i + c_0 + lc_0))

        derivative[1] = (xi - b * eta + m_i) / tau

        derivative[2] = (tau * (alpha - f_i * alpha ** 3 / 3.0 - beta) +
                    K21 * ((xi/ Cik) - alpha) +
                    tau * (II_i + c_0 + lc_0))

        derivative[3] = (alpha - b * beta + n_i) / tau
        """

        py_ast = ast.parse(
"""
derivative[0] = (tau * (xi - e_i * xi ** 3 / 3.0 - eta) +
        K11 * ((xi/ Aik) - xi) -
        K12 * ((alpha/ Bik) - xi) +
        tau * (IE_i + c_0 + lc_0))
"""
        )
        dfdag = nes.ast_to_dfdag(
                py_ast, 
                {
                    'xi':('n_nodes', 3),
                    'eta':('n_nodes', 3),
                    'alpha':('n_nodes', 3),
                    'beta':('n_nodes', 3),
                    'derivative' : (4, 'n_nodes', 3),
                    'lc_0' : ('n_nodes', 3),
                    'c_0' : ('n_nodes', 3),
                    'K11' : 'scalar',
                    'K12' : 'scalar',
                    'K21' : 'scalar',
                    'tau' : 'scalar',
                    'a' : 'scalar',
                    'b' : 'scalar',
                    'sigma' : 'scalar',
                    'mu' : 'scalar',
                    'Aik' : (3,),
                    'Bik' : (3,),
                    'Cik' : (3,),
                    'e_i' : (3,),
                    'f_i' : (3,),
                    'IE_i' : (3,),
                    'II_i' : (3,),
                    'm_i' : (3,),
                    'n_i' : (3,)
                    }
                )
        import ipdb; ipdb.set_trace()


                
    
class CodeGenTestold(unittest.TestCase):
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
        lin = dfdag.linearize()
        self.assertTrue(lin[1] == a5)
        self.assertTrue(lin.index(a2) > lin.index(a3))
        self.assertTrue(lin.index(a1) > lin.index(a4))
        self.assertTrue(lin.index(a1) > lin.index(a2))
        self.assertTrue(lin.index(a1) > lin.index(a3))
    
    def loop_block_grower_test(self):
        """
        digraph{
          op1->a
          op1->b
          op2->c
          op2->d
          op3->e->op1
          op3->f->op2
          dot->f
          op4->h
          op4->i->op3
          op5->j->op4
          op5->k->dot
        }
        """
        a = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        b = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        c = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        d = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        e = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        f = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        g = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        h = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        i = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        j = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        k = Value(type=ArrayType(data=ArrayData(shape=("nodes",)))) 
        l = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        op1 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [a,b], 
                e)
        op2 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [c,d], 
                f)
        op3 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [e,f], 
                i)
        op4 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [h,i], 
                j)
        op5 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [j,k], 
                l)
        dot = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes",)))
                ),
                [f,g], 
                k)
        dfdag = DFDAG([op1, op2, op3, op4, op5, dot],[a, b, c, d, e, f, g, h, i, j,k,l ])
        lbg = nes.LoopBlockGrower(dimension="nodes")
        lbg.visit(l)
        self.assertTrue(lbg.loop_block.applies == set([op1, op2, op3, op4, op5,dot]))
        lbg = nes.LoopBlockGrower(dimension="modes")
        lbg.visit(l)
        self.assertTrue(lbg.loop_block.applies == set([op1, op3, op4, op5 ]))



    def loop_block_test(self):
        """
        digraph {
          d->op1->a
          op1->b
          e->op2->b
          op2->c
          f->red->d
          red->e
          h->op3->f
          op3->g
          j->op4->h
          op4->i
        }
        """
        a = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        b = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        c = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        d = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        e = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        f = Value(type=ArrayType(data=ArrayData(shape=("nodes",))))
        g = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        h = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        i = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        j = Value(type=ArrayType(data=ArrayData(shape=("nodes","modes"))))
        op1 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [a,b], 
                d)
        op2 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [b,c], 
                e)
        op3 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [f,g], 
                h)
        op4 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes","modes")))
                ),
                [h,i], 
                j)
        dot = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes","modes"))),
                        ArrayType(data=ArrayData(shape=("nodes","modes")))],
                    ArrayType(data=ArrayData(shape=("nodes",)))
                ),
                [d,e], 
                f)
        dfdag = DFDAG([op1, op2, op3, op4, dot],[a, b, c, d, e, f, g, h, i, j])
        lb = nes.LoopBlocker(dimension="modes")
        lb.visit(j)
        self.assertTrue(len(lb.loop_blocks) == 3) # fusing only consecutive expressions for now
        self.assertTrue(lb.loop_blocks[0].applies == set([op4, op3]))
        self.assertTrue( (lb.loop_blocks[1].applies == set([op1]) and lb.loop_blocks[2].applies == set([op2]) ) or (lb.loop_blocks[2].applies == set([op1]) and lb.loop_blocks[1].applies == set([op2])) ) # quite ugly :(


    
    def value_dependecy_test(self):
        a = Value(type=ScalarType())
        b = Value(type=ScalarType())
        c = Value(type=ScalarType())
        d = Value(type=ScalarType())
        e = Value(type=ScalarType())
        f = Value(type=ScalarType())
        op1 = Apply(
                BinOp(ast.Add(),
                    [  ScalarType(),ScalarType()],
                     ScalarType()               ),
                [a,b], 
                c)
        op2 = Apply(
                BinOp(ast.Add(),
                    [  ScalarType(), ScalarType()],
                     ScalarType()               ),
                [b,d], 
                e)
        op3 = Apply(
                BinOp(ast.Add(),
                    [  ScalarType(),ScalarType()],
                     ScalarType()               ),
                [e,c], 
                f)
        dfdag = DFDAG([op1, op2, op3],[a, b, c, d, e, f])
        dc = nes.DependencyCollector()
        dc.visit(f)
        self.assertTrue(dc.value_deps.has_key(a))
        self.assertTrue(dc.value_deps.has_key(b))
        self.assertTrue(dc.value_deps.has_key(c))
        self.assertTrue(dc.value_deps.has_key(d))
        self.assertTrue(dc.value_deps.has_key(e))
        self.assertTrue(dc.value_deps.has_key(f))
        vd = {}
        vd[a] = set([op1])
        vd[b] = set([op1,op2])
        vd[c] = set([op3])
        vd[d] = set([op2])
        vd[e] = set([op3])
        vd[f] = set()
        self.assertTrue(dc.value_deps == vd)


    def simple_ctree_test(self):
        # single loop block (only nodes, modes later) 1 state variable
        # result should be single loop with scalar intermediate values, 
        # indexed array inputs and indexed array output
        # g = f *( a * b + d ) where f is scalar (other are input arrays)
        a = Value(type=ArrayType(data=ArrayData(shape=("nodes",))))
        b = Value(type=ArrayType(data=ArrayData(shape=("nodes",))))
        c = Value(type=ArrayType(data=ArrayData(shape=("nodes",))))
        d = Value(type=ScalarType())
        e = Value(type=ArrayType(data=ArrayData(shape=("nodes",))))
        f = Value(type=ScalarType())
        g = Value(type=ArrayType(data=ArrayData(shape=("nodes",))))
        op1 = Apply(
                BinOp(ast.Mult(),
                    [   ArrayType(data=ArrayData(shape=("nodes",))),
                        ArrayType(data=ArrayData(shape=("nodes",)))],
                    ArrayType(data=ArrayData(shape=("nodes",)))
                ),
                [a,b], 
                c)
        op2 = Apply(
                BinOp(ast.Add(),
                    [   ArrayType(data=ArrayData(shape=("nodes",))),
                        ScalarType()],
                    ArrayType(data=ArrayData(shape=("nodes",)))
                ),
                [c,d], 
                e)
        op3 = Apply(
                BinOp(ast.Mult(),
                    [   ArrayType(data=ArrayData(shape=("nodes",))),
                        ScalarType()],
                    ArrayType(data=ArrayData(shape=("nodes",)))
                ),
                [e,f], 
                g)
        dfdag = DFDAG([op1, op2, op3],[a, b, c, d, e, f, g])
        lb = LoopBlock("nodes")
        lb.applies = [op1, op2, op3]

        self.assertTrue(False) # write me
    def value_collector_test(self):
        self.assertTrue(False) # write me

