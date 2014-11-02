"""
Specializers for neural ensamble models. To be used together with TVB.
"""

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.visitors import NodeTransformer, NodeVisitor
from ctree.c.nodes import FunctionCall, CFile, Assign, ArrayRef, SymbolRef, Constant, Op, UnaryOp, Deref, For, Lt, PostInc
from ctree.cpp.nodes import CppInclude #TODO refactor to C?
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from ctree.templates.nodes import StringTemplate
from ctypes import CFUNCTYPE, c_double, c_int


import ast
from astmonkey import transformers as monkeytrans

import networkx as nx


import numpy as np

def MultiArrayRef(name, *idxs):
    """
    Given a string and a list of ints, produce the chain of
    array-ref expressions:

    >>> MultiArrayRef('foo', 1, 2, 3).codegen()
    'foo[1][2][3]'
    """
    tree = ArrayRef(SymbolRef(name), idxs[0])
    for idx in idxs[1:]:
        tree = ArrayRef(tree, Constant(idx))
    return tree


class CModelDfunFunction(ConcreteSpecializedFunction):
    def __init__(self, sim, pars):
        self.params = np.zeros(len(pars))
        for i in range(len(pars)):
            assert(len(getattr(sim.model, pars[i]))==1)#TODO why are those arrays anyway?
            self.params[i] = getattr(sim.model, pars[i])[0] 
        self.derivative = np.zeros((sim.model.nvar, sim.number_of_nodes, sim.model.number_of_modes))

    def finalize(self, program, tree, entry_name):
        self._c_function = self._compile(program, tree, entry_name)
    def __call__(self, state_variables, coupling, local_coupling):
        self._c_function(state_variables, coupling, local_coupling, self.params, self.derivative)
        return self.derivative


class CMathConversions(NodeTransformer):
    """
    Transforms Python operators and basic math functions to C math equivalents.
    """

    PY_OP_TO_CTREE_FN ={
            ast.Pow: "pow",
            ast.USub: Op.SubUnary
            # TODO log, exp, sqrt, ... see math.h
    }

    def visit_Module(self, node):
        node.body.insert(0, CppInclude(target="math.h"))
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if self.PY_OP_TO_CTREE_FN.has_key(type(node.op)):
            rhs = self.visit(node.operand)
            op = self.PY_OP_TO_CTREE_FN[type(node.op)]()
            return UnaryOp(op, rhs) 
        else:
            return self.generic_visit(node)
    def visit_BinOp(self, node):
        if self.PY_OP_TO_CTREE_FN.has_key(type(node.op)):
            lhs = self.visit(node.left)
            rhs = self.visit(node.right)
            fn = self.PY_OP_TO_CTREE_FN[type(node.op)]
            return FunctionCall(fn, [lhs, rhs]) #TODO does this work for anything else but pow?
        else:
            return self.generic_visit(node)
    def visit_Call(self,node):
        #TODO add more as needed
        if isinstance(node.func, ast.Attribute) and node.func.value.id == 'numpy':
            if node.func.attr=='exp': 
                a1 = self.visit(node.args[0])
                return FunctionCall( "exp", [a1])
        return self.generic_visit(node)


class  DotReplacement(NodeTransformer):
    def visit_Call(self,node):
        # both are reductions over variables or modes, should produce loops
        # currently only direct refernces supported, no temporaries
        if isinstance(node.func, ast.Attribute) and node.func.value.id == 'numpy':
            if node.func.attr=='dot': 
                a1 = self.visit(node.args[0])
                a2 = self.visit(node.args[1])
                return FunctionCall( "dot", [a1, a2])
            elif node.func.attr == 'sum':
                # expects direct refernces, allowed only over modes ()
                # TODO automatically generate intermediate temporary results
                # fixed variable, per node, sum over mode dimension
                assert(node.keywords[0].arg == "axis" and node.keywords[0].value.n == 1) 
                return FunctionCall( "sum", [SymbolRef(node.args[0].id)])
            elif node.func.attr == 'array':
                return self.generic_visit(node)
            else:
                raise RuntimeError('Unknown numpy call')
        else:
            return self.generic_visit(node)


class ModelParameters(NodeTransformer):
    '''
    Both changes the syntax and accumulates the list of used parameters.
    '''

    def __init__(self):
        self.pars = {} 

    def visit_Attribute(self, node):
        if node.value.id == 'self':
            if not self.pars.has_key(node.attr):
                self.pars[node.attr] = len(self.pars)
            return ArrayRef("pars", self.pars[node.attr])
        else:
            return self.generic_visit(node)

class NamesToArrays(NodeTransformer):
    
    def __init__(self, array_type):
        self.array_type = array_type

    # get rid of model parameters
    def visit_Attribute(self, node):
        if node.value.id == 'numpy':
            # not a vairable, skip
            return self.generic_visit(node)
        else:
            raise RuntimeError('Unsupported structure for attributes: %s'% node.value.id)

    def visit_Subscript(self, node):
        # get rid of newaxis syntactic sugar
        for s in node.slice.dims:
            if isinstance(s, ast.Index) and isinstance(s.value, ast.Attribute) and s.value.attr == 'newaxis':
                return node.value
        if isinstance(node.parent, ast.Assign):
            assert(isinstance(node.slice.dims[0], ast.Index))
            return MultiArrayRef(SymbolRef(node.value.id),SymbolRef(node.slice.dims[0].value.n), SymbolRef("node_it"), SymbolRef("mode_it"))
        else:
            raise RuntimeError("Don't know what to do with subscript.")

    def visit_Assign(self,node):
        if not (isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'derivative'):
            return self.generic_visit(node)
        
        return Assign(
                MultiArrayRef(
                    SymbolRef('derivative'),
                    Constant(0),
                    SymbolRef("node_it"),
                    SymbolRef("mode_it")
                ),
                SymbolRef(node.value.args[0].elts[0].id)
                )

    def visit_Return(self,node):
        assert(node.value.id == 'derivative') 
        return None

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Param):
            # function declaration
            if node.id == 'self':
                # C is selfless
                return None
            else:
                return SymbolRef(Deref(Deref(Deref(node.id))), self.array_type._dtype_.type())
        elif isinstance(node.parent, ast.Assign) and node.parent_field == 'targets':
            # local variable
            return SymbolRef(node.id, self.array_type._dtype_.type()) # danger of repeated declarations!
        else:
            return self.generic_visit(node)#SymbolRef(node.id, self.array_type())

class DfunDef(NodeTransformer):
    def __init__(self,pars):
        self.pars = pars


    def visit_FunctionDef(self,node):
        node.args.args.append(
                SymbolRef(
                    Deref('pars'),
                    c_double()
                    )
                )
        node.args.args.append(
                SymbolRef(
                    'n_nodes',
                    c_int()
                    )
                )
        node.args.args.append(
                SymbolRef(
                    'n_modes',
                    c_int()
                    )
                )
        node.args.args.append(
                SymbolRef(
                    Deref(Deref(Deref('derivative'))),
                    c_double()
                    )
                )
        return node

class LoopBody(NodeTransformer):

    def visit_FunctionDef(self,node):
        loop = [For(
                init = Assign(SymbolRef("node_it", c_int()), Constant(0)),
                test = Lt(SymbolRef("node_it"), Constant( SymbolRef('n_nodes'))),
                incr = PostInc(SymbolRef("node_it")),
                body = [
                    For(
                        init = Assign(SymbolRef("mode_it", c_int()), Constant(0)),
                        test = Lt(SymbolRef("mode_it"), Constant(SymbolRef('n_modes'))),
                        incr = PostInc(SymbolRef("mode_it")),
                        body = node.body
                        )
                    ]
                )]
        node.body = loop
        return node

class RemoveComments(NodeTransformer):
    # triple quoted strings are expressions (no-return statements)... sigh
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Str):
            # gone it is
            return None
        else:
            return self.generic_visit(node)

class DataDependencies(NodeVisitor):
    def __init__(self):
        self.dag = nx.DiGraph()
        self.variables = {}

    def write_dag(self, filename):
        for node in self.dag.nodes_iter():
            self.dag.node[node]['label'] = ast.dump(node)

        nx.write_graphml(self.dag,filename)

    def visit_FunctionDef(self,node):
        for arg in node.args.args:
            self.dag.add_node(arg)
            self.variables[arg.id] = arg
        self.generic_visit(node)

    #def visit_Assign(self,node):
    #    assert( len(node.targets)==1) # for now...
    #    target = node.targets[0]

    #    self.dag.add_edge(target, node.value)
    #    self.variables[target.id] = node.value 
    #    self.generic_visit(node)

    def visit_Subscript(self,node):
        if isinstance(node.value, ast.Name):
            self.dag.add_edge(node, self.variables[node.value.id])
        else:
            self.dag.add_edge(node, node.value)
        self.generic_visit(node)

    def visit_BinOp(self,node):
        if isinstance(node.left, ast.Name):
            self.dag.add_edge(node, self.variables[node.left.id])
        else:
            self.dag.add_edge(node, node.left)
        if isinstance(node.right, ast.Name):
            self.dag.add_edge(node, self.variables[node.right.id])
        else:
            self.dag.add_edge(node, node.right)
        self.generic_visit(node)

    def visit_UnaryOp(self,node):
        if isinstance(node.operand, ast.Name):
            self.dag.add_edge(node, self.variables[node.operand.id])
        else:
            self.dag.add_edge(node, node.operand)
        self.generic_visit(node)

    def visit_Call(self,node):
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self.dag.add_edge(node,self.variables[arg.id])
            else:
                self.dag.add_edge(node,arg)
        self.generic_visit(node)

    def visit_Name(self,node):
        if isinstance(node.ctx, ast.Store):
            self.dag.add_edge(node, node.parent.value)
            self.variables[node.id] = node 
            self.generic_visit(node)

        ## handeled by function def
        #elif isinstance(node.ctx, ast.Param): 
        #    self.variables[node.id] = node
        #    self.dag.add_node(node)
        ## should be covered by ops
        #elif isinstance(node.ctx, ast.Load) and not isinstance(node.parent, ast.Attribute):
        #    self.dag.add_edge(node.parent, self.variables[node.id])

        self.generic_visit(node)

def deps_to_types(deps, sim):
    G = deps.reverse()
    sources = [n for n,d in G.in_degree_iter() if d ==0]
    for s in sources:
        print ast.dump(s)
    import ipdb; ipdb.set_trace()
    # type function params, (also self.params)
    # add common source to function params
    node_types = {}
    # dfs propagate types
    #   per element array ops
    #   per element array op scalar
    #   slices

    return node_types


class CModelDfun(LazySpecializedFunction):
    def __init__(self,py_ast, sim):
        super(CModelDfun,self).__init__(py_ast)
        self.sim = sim

    def args_to_subconfig(self, args):
        """
        Analyze arguments and return a 'subconfig', a hashable object
        that classifies them. Arguments with identical subconfigs
        might be processed by the same generated code.
        """
        state_variables = args[0]
        coupling = args[1]
        local_coupling = args[2]

        arg_config = {
                'state_vars' : np.ctypeslib.ndpointer(
                    state_variables.dtype,
                    state_variables.ndim, 
                    state_variables.shape),
                'coupling' : np.ctypeslib.ndpointer(
                    coupling.dtype,
                    coupling.ndim, 
                    coupling.shape),
                'local_coupling' : np.ctypeslib.ndpointer(
                    local_coupling.dtype,
                    local_coupling.ndim, 
                    local_coupling.shape),
                }
        #TODO: this only captures problem size, should also hash the model type...
        return arg_config
    

    def transform(self, tree, program_config):
        arg_config, tuner_config = program_config
        state_vars = arg_config['state_vars']
        
        tree = monkeytrans.ParentNodeTransformer().visit(tree)
        tree = RemoveComments().visit(tree)
        datadep = DataDependencies()
        datadep.visit(tree)

        deps_to_types(datadep.dag, self.sim)
        import ipdb; ipdb.set_trace()

        # traverse the python AST, replace numpy slices address 
        # locate dot operations and replace accordingly
        tree = CMathConversions().visit(tree)
        tree = DotReplacement().visit(tree)
        params = ModelParameters()
        tree = params.visit(tree)
        pars = sorted(params.pars, key=params.pars.get)

        tree = NamesToArrays(state_vars).visit(tree)
        tree = DfunDef(pars).visit(tree)
        tree = LoopBody().visit(tree)
        # not needed, for now...


        
        prog_tree = CFile("generated_dfun", 
            tree.body #dfun definition and includes
            )
        prog_tree = PyBasicConversions().visit(prog_tree)
        proj = Project([prog_tree])


        fn = CModelDfunFunction(self.sim, pars)
        entry_point_typesig = CFUNCTYPE(None, state_vars)

        return fn.finalize("dfun", proj, entry_point_typesig)

