"""
Specializers for neural ensamble models. To be used together with TVB.
"""

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.visitors import NodeTransformer, NodeVisitor
import  ctree.c.nodes as ctcn #FunctionCall, CFile, Assign, ArrayRef, SymbolRef, Constant, Op, UnaryOp, Deref, For, Lt, PostInc, BinaryOp
from ctree.cpp.nodes import CppInclude #TODO refactor to C?
from ctree.nodes import Project
import  ctree.transformations as ctt
from ctree.templates.nodes import StringTemplate
from ctypes import CFUNCTYPE, c_double, c_int
from ctree.types import get_ctype

import ctypes


import ast
from astmonkey import transformers as monkeytrans

import networkx as nx


import numpy as np


import dfdag, usr

class UseDefs:
    """
    This class encapsulates the logic required to track the use-def locations
    of the contents of arrays. Internal state is updated during parsing with
    every use and def, so it allows us to correctly represent the semantics of
    the original code.
    """
    def __init__(self):
        # This data structure holds the reaching definitions of array variables
        # during parsing. Main index are the instances of ArrayData
        # (representing the particular array in memory), which are shared by
        # all references represented by ArrayType instances. For every
        # ArrayData, we hold the list of all reaching definitions given by a
        # tuple (value,usr), where value is a Value node in the df-DAG (place
        # of definition) and usr is USR instance describing the extent of the
        # definition (can change with following partial kills).
        # {array_data: [(value_node, usr), ...], ...}
        self._array_defs = {}

        # This data structure holds the uses of curently valid contents of the
        # array (see _array_defs above). We consider application of an
        # operation on an array as an use, excluding "supporting operations",
        # such as broadcast, create_view, barrier, etc. Every kill has to
        # synchronize with the operations, which use the invalidated data. For
        # every definiton, we use the resulting Value node as a key, and  we
        # keep the list of pairs (value, use) where the value is the result of
        # an operation using the defined data, and USR extent of the use. 
        # {value_node: [(value_node, usr), ...], ...}
        self._array_uses = {}

    def define(self, value_node):
        '''
        Defines the contents of an ArrayData to the extent given by the
        ArrayType slice. Returns list of value nodes, which need to be guarded
        by a barrier before the kill can take place. 
        '''
        def_usr = self._array_to_usr(value_node.type)
        uses = set()
        new_defs = []

        #this seriously needs review
        if self._array_defs.has_key(value_node.type.data):
            for old_def in self._array_defs[value_node.type.data]:
                if old_def[1].intersect(def_usr).is_empty():
                    new_defs.append(old_def)
                else:
                    conflict = def_usr.intersect(old_def[1])

                    if self._array_uses.has_key(old_def[0]):
                        for use in self._array_uses[old_def[0]]:
                            if not use[1].intersect(conflict).is_empty():
                                uses.add(use[0]) 

                    old_def = list(old_def)
                    old_def[1] = old_def[1].complement(conflict)
                    new_defs.append(tuple(old_def))
        new_defs.append( (value_node, def_usr) )
        self._array_defs[value_node.type.data] = new_defs

        return list(uses)
        
    def use(self, in_value_node, out_value_node):
        '''
        Checks the currently valid definitions and returns a list of value
        nodes contributing the required portion of the array. Also registers
        the use of the definitions for later checks on rewrites. 

        in_value_node: defines the array to be accessed and the slice

        out_value_node: result value of the operation: will enter the barriers
        '''

        use_usr = self._array_to_usr(in_value_node.type)
        defs = []

        for def_pair in self._array_defs[in_value_node.type.data]:
            usage = def_pair[1].intersect(use_usr)
            if not usage.is_empty():
                defs.append(def_pair[0])
                # register the use
                if self._array_uses.has_key(def_pair[0]):
                    self._array_uses[def_pair[0]].append((out_value_node, usage))
                else:
                    self._array_uses[def_pair[0]] = [ (out_value_node, usage) ]

        return defs

        

    def _array_to_usr(self, array):
        subscripts = []
        for i, dim in enumerate(array.data.shape):
            if isinstance(dim, int):
                if array.slice[i] == ":":
                    subscripts.append(dim) # we take this whole dimension
                else:
                    # assumes simple slice in this dimension 
                    assert(isinstance(array.slice[i],int))
                    subscripts.append((array.slice[i],array.slice[i]+1)) 
            else:
                # we track only known-sized dimensions
                pass
        return usr.USR(subscripts)


class DFValueNodeCreator(NodeVisitor):
    '''
    This visitor transforms the python AST to df-DAG.
    '''


    def __init__(self, shapes):
        self._value_map = {}
        self.applies = []
        self.result = None  # One result per DAG. Can be generalized in future.
        self.input_values = {}
        self.dfdag = None
        self._variable_map = {}
        self.usedefs = UseDefs()

        

        for var in shapes:
            if shapes[var] == 'scalar':
                self._variable_map[var] = dfdag.Value(type=dfdag.ScalarType())
            else:
                data = dfdag.ArrayData(shape=shapes[var]) 
                value = dfdag.Value(type=dfdag.ArrayType(data=data))
                self._variable_map[var] = value 
                self.usedefs.define(value)
            
            # because the map will change during parsing
            self.input_values[var] = self._variable_map[var] 

    
    def createDAG(self):
        values = set(self._value_map.values())
        for appl in self.applies:
            for inp in appl.inputs:
                values.add(inp)
            values.add(appl.output)
        values = list(values)
        
        return dfdag.DFDAG(self.applies, values, self.input_values, self.result)

    def _synchronize(self, defs, dest):
        '''
        Takes the collected definitions, creates a synchronization Apply node,
        and returns the resulting value node.
        '''

        # poor man's copy constructor -- TODO refactor to dfdag
        ins = []
        ins.extend(defs)
        new_inp = dfdag.Value() 
        new_inp.type = dfdag.ArrayType( data = dest.type.data, slice=dest.type.slice)
        sync = dfdag.Apply(dfdag.Synchronize(), ins, new_inp)
        self.applies.append(sync)
        return new_inp

    def visit_Assign(self,node):
        self.generic_visit(node)
        if len(node.targets ) > 1:
            raise NotImplementedError("Only single value return statements supported.")
        target = node.targets[0]

        # what we get from rhs
        val = self._value_map[node.value]

        if isinstance(target, ast.Subscript):
            # possibly incomplete kill

            # value corresponding to the synchronized Subscript
            lhs_val = self._value_map[target] 
            killed_defs = self.usedefs.define(lhs_val)

            
            syncval = self._synchronize(killed_defs, lhs_val)

            bcast = dfdag.Apply(dfdag.Broadcast(), [syncval,val], lhs_val)
            self.applies.append(bcast)

            self._value_map[node] = lhs_val # do we need this at all?
            #self._variable_map[target.value.id] = lhs_val # is this useful? Is this needed?
        elif isinstance(target, ast.Name):
            # complete kill
            if isinstance(val.type, dfdag.ArrayType):
                self.usedefs.define(val)
            self._variable_map[target.id] = val
            self._value_map[target] = val
        else:
            # E.g. no attributes. Not now, not later.
            raise NotImplementedError()

        
    def visit_AugAssign(self, node):
        # inplace operators, implement later if needed
        raise NotImplementedError()


    def visit_BinOp(self, node):
        self.generic_visit(node)
        inputs = []
        input_types = []
        for operand in [node.left, node.right]:
            inputs.append( self._value_map[operand] )
            input_types.append( self._value_map[operand].type )

        output = dfdag.Value()
        out_type = dfdag.ScalarType()
        if isinstance(inputs[0].type, dfdag.ArrayType):
            ins = self.usedefs.use(inputs[0],output)
            if len(ins) > 1:
                inputs[0] = self._synchronize(ins,inputs[0])
            out_type = inputs[0].type.broadcast_with(inputs[1].type)
        if isinstance(inputs[1].type, dfdag.ArrayType):
            ins = self.usedefs.use(inputs[1],output)
            if len(ins) > 1:
                inputs[1] = self._synchronize(ins,inputs[1])
            if isinstance(out_type,dfdag.ScalarType): # in case the first operand is scalar
                out_type = inputs[1].type.broadcast_with(inputs[0].type)

        output.type=out_type
        if isinstance(out_type, dfdag.ArrayType):
            self.usedefs.define(output) # binary operator always creates a copy

        self._value_map[node] = output
        
        routine = dfdag.BinOp(
                operator = node.op, 
                input_types = input_types,
                output_type = out_type) 
        app = dfdag.Apply(routine, inputs, output)
        self.applies.append(app)
            

    def visit_Subscript(self, node):
        #expects the subscripted value to have known type

        # no expressions allowed in the slice
        self.visit(node.value)
 
        sval = self._value_map[node.value]
        slice_shape = list(sval.type.shape)
        
        if isinstance(node.slice, ast.ExtSlice):
            #expect things like x[0,:,2]
            for i, dim in enumerate(node.slice.dims):
                if isinstance(dim, ast.Index):
                    slice_shape[i] = dim.value.n
                else:
                    assert( dim.lower is None and dim.upper is None and dim.step is None)
                    # for now, could be generalized using lower/upper/step
                    slice_shape[i] = ":" 
            for i in range(len(node.slice.dims), len(slice_shape)):
                # cover ":" spreading over multiple tailing dimensions
                slice_shape[i] = ":"
            

        elif isinstance(node.slice, ast.Index):
            # e.g. x[3]
            slice_shape[0] = node.slice.value.n
            for i in range(1,len(slice_shape)):
                slice_shape[i] = ":"
        else:
            # do we need something like x[:] => ast.Slice?
            raise NotImplementedError()
        slice = tuple(slice_shape)

        newval = dfdag.Value()
        # poor man's copy constructor
        newval.type = dfdag.ArrayType( data = sval.type.data, slice=sval.type.slice)
        # newval.type.data = sval.type.data #TODO Wut? Just remove this.
        newval.type.apply_slice(slice)
        self._value_map[node] = newval


    def visit_Name(self, node):
        if self._variable_map.has_key(node.id):
            self._value_map[node] = self._variable_map[node.id]
        else:
            #value = dfdag.Value(self.variable_types.get(node.id,None))
            value = dfdag.Value(None)
            self._value_map[node] = value
            self._variable_map[node.id] = value

    def visit_Num(self, node):
        value = dfdag.Value(type=dfdag.Constant(node.n))
        self._value_map[node] = value

    def visit_Return(self, node):
        self.generic_visit(node)
        sval = self._value_map[node.value]
        if isinstance(sval, dfdag.ArrayType):
            syncvals = self.usedefs.use(sval,None)
            if len(syncvals) > 1:
                sval = self._synchronize(syncvals,sval)
        #ret = dfdag.Apply(dfdag.Return(), [sval], None) 
        self.result = sval

class DFDAGVisitor(ast.NodeVisitor):
    """
    Walks the graph upwards following the dependencies.
    """
    def __init__(self):
        self._visited = set() # DAG is not a tree

    def generic_visit(self, node):
        if node not in self._visited:
            self._visited.add(node)
            for dep in node.depends():
                self.visit(dep)
        else:
            pass # been there, done that


class BFSVisitor(ast.NodeVisitor):
    def __init__(self):
        self._queue = [] # poor mans queues: uses pop(0) and append(node)
        self._visited = set() # again, DAG is not a tree

    def generic_visit(self,node):
        if node not in self._visited:
            self._visited.add(node)
            for dep in node.depends():
                self._queue.append(dep)
        else:
            pass # been there, done that
        if len(self._queue) > 0:
            self.visit(self._queue.pop(0))

class SubGraphCollector(DFDAGVisitor):
    def __init__(self):
        self.values = set()
        self.applies = set()
        super(SubGraphCollector,self).__init__()

    def visit_Value(self, node):
        self.generic_visit(node)
        self.values.add(node)
    def visit_Apply(self, node):
        self.generic_visit(node)
        self.applies.add(node)

class LoopBlocker(BFSVisitor): 
    """
    loop fusion pre-linearizaion
    """
    
    def __init__(self, dimension):
        self.dim = dimension
        self.loop_blocks = []
        self._blocked = set()
        super(LoopBlocker,self).__init__()

    def visit_Value(self, node):
        if isinstance(node.type, dfdag.ArrayType) and self.dim in node.type.shape:
            if node.source is not None and node.source not in self._blocked:
                # engage blocking
                lbg = LoopBlockGrower(self.dim, set(self._blocked))
                lbg.visit(node)
                self.loop_blocks.append(lbg.loop_block)
                self._blocked.update(lbg.loop_block.applies)
            else:
                pass # already part of block, passing by
        else:
            # invariant
            pass
        self.generic_visit(node)
            
        
class LoopBlockGrower(BFSVisitor):
    """
    Give me seed, I'll grow you a block.
    """
    def __init__(self, dimension, forbidden=set()):
        self.dim = dimension
        self._forbidden = set(forbidden)
        self.loop_block = dfdag.LoopBlock(self.dim)
        super(LoopBlockGrower,self).__init__()

    def visit_Value(self,node):
        if node not in self._forbidden:
            if isinstance(node.type, dfdag.ArrayType) and self.dim in node.type.shape:
                # previously unvisited, not forbidden, safe to add source
                if node.source is not None:
                    self.loop_block.applies.add(node.source)
            else:
                # invariant, forbid underlaying DAG (remove already placed applies?)
                sg = SubGraphCollector()
                sg.visit(node)
                self._forbidden.update(sg.values)
                self.loop_block.applies.difference_update(sg.applies)
        else:
            pass # you shall not pass
        self.generic_visit(node) # hmm...


class DependencyCollector(DFDAGVisitor):
    def __init__(self):
        self.value_deps = {}
        super(DependencyCollector,self).__init__()

    def visit_Apply(self,node):
        for val in node.inputs:
            if not self.value_deps.has_key(val):
                self.value_deps[val] = set()
            if node not in self.value_deps[val]:
                self.value_deps[val].add(node)
        self.generic_visit(node)

    def visit_Value(self, node):
        if not self.value_deps.has_key(node):
            self.value_deps[node] = set()
        self.generic_visit(node)


# TODO refactor this to DFValueNodeCreator and rename to dfdag creator
def ast_to_dfdag(py_ast, variable_shapes={}):
    tree = monkeytrans.ParentNodeTransformer().visit(py_ast)
    tree = RemoveComments().visit(py_ast)

    dvn = DFValueNodeCreator(variable_shapes)
    dvn.visit(py_ast)
    return dvn.createDAG()

class DFDAGTopowalker(ast.NodeVisitor):
    '''
    Walks the toplogical sort of a df-DAG.
    '''
    def __init__(self, dfdag):
        self._dfdag = dfdag

    def walk(self):
        linearization = reversed(self._dfdag.linearize())
        for node in linearization:
            self.visit(node)

    def generic_visit(self, node):
        # there is nothing to do, next!
        pass

class CtreeBuilder(DFDAGTopowalker):
    '''
    Walks the linearization and builds a Ctree representation from the Apply nodes. Makes sure every needed 
    '''
    

    def __init__(self, df_dag):
        super(CtreeBuilder,self).__init__(df_dag)

        '''
        Following two dictionaries keeps the mapping from Value nodes to
        symbols in the generated code. The scalar symbols are indexed by the
        Value node and are singular {Value: string}, whereas the array symbols
        are indexed by ArrayData and multiple symbols can correspond to single
        ArrayData {ArrayData: [string, string, ...]}. The symbols are unique
        across both types.

        The dimension symbols is a table of known variables describing the
        problem-instance dependent sizes of particular arrays. {string:string}
        '''
        self.scalar_symbols = {}
        self.array_symbols = {}
        self.dimension_symbols = {}
        self.c_ast = ctcn.CFile()
        # register inputs
        for in_sym in self._dfdag.input_values:
            in_val = self._dfdag.input_values[in_sym]
            if isinstance(in_val.type, dfdag.ScalarType):
                in_cs = ctcn.SymbolRef.unique("s", ctypes.c_double())
                self.scalar_symbols[in_val] = in_cs
            else:
                assert(isinstance(in_val.type, dfdag.ArrayType))
                in_cs = ctcn.SymbolRef.unique("a",ctypes.POINTER(ctypes.c_double)())
                self.array_symbols[in_val.type.data] = in_cs
                for dim in in_val.type.data.shape:
                    if isinstance(dim, str) and not self.dimension_symbols.has_key(dim):
                        self.dimension_symbols[dim] = ctcn.SymbolRef.unique("d", ctypes.c_long())
                        self.c_ast.body.append(
                                ctcn.Assign(
                                    self.dimension_symbols[dim], 
                                    ctcn.SymbolRef(dim)
                                    )
                                )
            self.c_ast.body.append(
                    ctcn.Assign(
                        in_cs, 
                        ctcn.SymbolRef(in_sym)
                        )
                    )

        # register result of the code block (return value)
        if self._dfdag.result is not None:
            if isinstance(self._dfdag.result.type, dfdag.ScalarType):
                # TODO, should be returned? Don't care about this one yet.
                raise NotImplementedError("Scalar results are not implemented.")
            else:
                assert(isinstance(self._dfdag.result.type, dfdag.ArrayType))
                res_cs = ctcn.SymbolRef.unique("a",ctypes.POINTER(ctypes.c_double)())
                self.array_symbols[self._dfdag.result.type.data] = res_cs
                self.return_symbol = res_cs # to be used in function declaration



    #
    #   Routines
    #
    def _translate_routine(self, routine, inputs, output):
        method = '_translate_' + routine.__class__.__name__
        translate = getattr(self, method) # exception for not implemented  
        return translate(routine, inputs, output)
    
    def _translate_BinOp(self,routine, inputs, output):
        op = ctt.PyBasicConversions.PY_OP_TO_CTREE_OP[type(routine.operator)]()
        return ctcn.Assign(output, ctcn.BinaryOp(inputs[0], op, inputs[1]))
    # TODO other routines, such as numpy functions


    # 
    #    Values
    # 
    def _translate_value(self, value):
        # constants, scalars and whole array references
        method = '_translate_' + value.type.__class__.__name__
        translate = getattr(self, method) # exception for not implemented  
        return translate(value)

    def _translate_ScalarType(self,value):
        return self.scalar_symbols[value].copy()

    def _translate_Constant(self,value):
        return ctcn.Constant(value.type.number)

    def _translate_ArrayType(self,value):
        # this should be properly indexed according to loop context
        if self._index_map is not None:
            # 
            return self._add_indexing(value.type)
        else:
            return ctcn.array_symbols[value.type.data].copy()

    #
    #   Indexed values. We assume to recieve correct broadcasting shapes here.
    #
    def _shape_size_to_symbol(self, dim):
        if isinstance(dim, str):
            return ctcn.SymbolRef(self.dimension_symbols[dim].copy())
        else:
            assert isinstance(dim, int)
            return ctcn.Constant(dim)
    

    def _allocate_array(self, array_data):
        # symbol should be pointer declaration...
        sym = ctcn.SymbolRef.unique("a",ctypes.POINTER(ctypes.c_double)())
        self.array_symbols[array_data] = sym.copy()

        array_size = ctcn.SizeOf(sym.copy() )
        for dim in array_data.shape:
            array_size = ctcn.Mul(self._shape_size_to_symbol(dim),array_size)

        malloc_ast = ctcn.Cast(
                sym_type=ctypes.POINTER(ctypes.c_double)(), 
                value=ctcn.FunctionCall(
                    'malloc',
                    [array_size]
                    )
                )

        return ctcn.Assign(sym, malloc_ast)

    
    def _add_indexing(self, array_type):
        """
        Takes an array (defined by array_type) and applies indexing from loop
        nest context given by index_map. The index map can be longer than shape
        of the array because of numpy broadcasts. 
        """
        # this seriously needs review

        summands = []
        shape_dim = 0
        for i, j in enumerate(range(len(self._index_map)-len(array_type.slice),len(self._index_map) )):
            if isinstance(array_type.slice[i], int):
                # constant slice offset
                mul = ctcn.Constant(array_type.slice[i]) 
            else:
                # iteration dimension
                mul = self._index_map[j][0].copy()

            # multiply by sizes of trailing dimensions
            for k in range(i+1,len(array_type.data.shape)):
                mul = ctcn.Mul(mul, self._shape_size_to_symbol(array_type.data.shape[k]))
            summands.append(mul)    

        #sum things up
        index = summands[0] # there better be something...
        for i in range(1,len(summands)):
            index = ctcn.Add(summands[i], index)

        return ctcn.ArrayRef(self.array_symbols[array_type.data].copy(), index)

    def _create_loop_nest(self, routine, in_cs, out_cs):
        """
            Derives the number of loops and indices from output shape and wraps
            the given loop nest_body.
        """

        body_c_ast = self._translate_routine(routine, in_cs, out_cs)
        for dim, size in reversed(self._index_map):
            body_c_ast = ctcn.For(
                    ctcn.Assign(dim, ctcn.Constant(0)),
                    ctcn.Lt(dim.copy(), size),
                    ctcn.PostInc(dim.copy()), 
                    [body_c_ast]
                    )
        return body_c_ast


    def visit_Apply(self, node):
        if isinstance(node.routine, dfdag.Synchronize):
            # just a helper routine, ignore
            return
        if isinstance(node.routine, dfdag.Return):
            raise NotImplementedError("TODO: handling return statements.")
        
        
        if isinstance(node.output.type, dfdag.ScalarType):
            self._index_map = None
            in_cs = []
            for input in node.inputs:
                in_cs.append( self._translate_value(input) )
            out_cs = ctcn.SymbolRef.unique("s", ctypes.c_double())
            self.scalar_symbols[node.output] = out_cs.copy()
            
            c_ast = self._translate_routine(node.routine, in_cs, out_cs)
            self.c_ast.body.append( c_ast ) 
        else:
            #just in case...
            assert( isinstance(node.output.type, dfdag.ArrayType) )

            if not self.array_symbols.has_key(node.output.type.data):
                # allocate and create symbol
                self.c_ast.body.append( self._allocate_array(node.output.type.data))
            
            self._index_map = [] # this is very important, it changes behaviour of other functions
            for dim in node.output.type.shape:
                self._index_map.append(
                        (   ctcn.SymbolRef.unique("i",ctypes.c_long()), 
                            self._shape_size_to_symbol(dim)
                            )
                        )

            out_cs = self._translate_value(node.output)
            in_cs = []
            for input in node.inputs:
                in_cs.append( self._translate_value(input) )

            loop_body_c_ast = self._create_loop_nest(
                    node.routine,
                    in_cs, 
                    out_cs)

            self.c_ast.body.append( loop_body_c_ast ) 

def dfdag_to_ctree(dfdag):
    ct_builder = CtreeBuilder(dfdag)
    ct_builder.walk()
    return ct_builder


def loop_block_ctree(loop_block, value_deps, value_variable_map):
    # linearize
    # allocate memory for new output variables and associate to array value data types
    # replace applies with assign
    # if value is not associated with variable, create def
    raise NotImplementedError("We will need this for sure.")





# cut and remove here
# -----------------------------
#






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
        self.n_nodes = sim.number_of_nodes
        self.n_modes = sim.model.number_of_modes

    def finalize(self, program, tree, entry_name):
        self._c_function = self._compile(program, tree, entry_name)
        return self

    def __call__(self, state_variables, coupling, local_coupling):
        self._c_function(state_variables, coupling, local_coupling, self.params, self.n_nodes, self.n_modes, self.derivative)
        return self.derivative


class CMathConversions(NodeTransformer):
    """
    Transforms Python operators and basic math functions to C math equivalents.
    """

    PY_OP_TO_CTREE_FN ={
            ast.Pow: "pow",
            ast.USub: ctcn.Op.SubUnary
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
    Change later for vector parameters with per-node values
    '''

    def __init__(self):
        self.pars = {} 

    def visit_Attribute(self, node):
        if node.value.id == 'self':
            if not self.pars.has_key(node.attr):
                self.pars[node.attr] = len(self.pars)
            return ArrayRef(SymbolRef('pars'), Constant(self.pars[node.attr]))
        else:
            return self.generic_visit(node)

class NamesToArrays(NodeTransformer):
    """
    Translates broadcasts to array refernces -- needs refactoring and generalization.
    """

    
    def __init__(self, array_type, node_types, variables):
        self.array_type = get_ctype(array_type._dtype_.type())
        self.node_types = node_types
        self.variables = variables


    # get rid of model parameters
    def visit_Attribute(self, node):
        if node.value.id != 'self':
            # not a vairable, skip
            return self.generic_visit(node)
        else:
            raise RuntimeError('Unsupported structure for attributes: %s'% node.value.id)




    def visit_Subscript(self, node):
        """
        translates numpy subscripts to pointer arithmetics...
        """
        assert(self.node_types[node][0] == "scalar") # just don't try anything funny
        assert(isinstance(node.slice,ast.ExtSlice) and isinstance(node.slice.dims[0].value, ast.Num))
        # TODO deal with modes...
        #return ArrayRef(node.value.id, str(node.slice.dims[0].value.n) + " * n_nodes + node_it" ) 
        return BinaryOp(
                SymbolRef(node.value.id), 
                Op.Add(),
                #test = Lt(SymbolRef("node_it"), Constant( SymbolRef('n_nodes'))),
                BinaryOp(
                    Constant(node.slice.dims[0].value.n),
                    Op.Mul(),
                    SymbolRef('n_nodes')
                    )
                )
        



    def visit_Assign(self,node):
        assert(len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)) 
        target = node.targets[0].id

        if isinstance(node.value,ast.Subscript):
            # special case
            return Assign(
                    #Deref(SymbolRef(target, self.array_type)),
                    SymbolRef(target, ctypes.POINTER( ctypes.c_double )()), 
                    self.visit(node.value)
                    )
        # looking for assignment derivative = numpy.array([...])
        if not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'array'):
            return self.generic_visit(node)
        # this needs to be done separately for each partial derivative
        new_nodes = []
        assert( len(node.value.args) == 1 and isinstance(node.value.args[0], ast.List))
        for i, var in enumerate(node.value.args[0].elts):
            assert(isinstance(var,ast.Name))
            new_nodes.append(
                    Assign(
                        ArrayRef(SymbolRef(target),
                            #str(i) + " * n_nodes + node_it" ), # this is getting a little verbatim...
                            BinaryOp(
                                BinaryOp(
                                    Constant(i),
                                    Op.Mul(),
                                    SymbolRef('n_nodes')
                                    ),
                                Op.Add(),
                                SymbolRef('node_it')
                                ),
                            ),
                        SymbolRef(var.id)
                        )
                    )
        return new_nodes

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Param):
            # function declaration
            if node.id == 'self':
                # C is selfless
                return None
            else:
                #return Deref(SymbolRef(node.id, self.array_type))
                return SymbolRef(node.id, ctypes.POINTER(ctypes.c_double)() )
        type_variable = self.variables[node.id]
        if isinstance(node.parent, ast.Assign) and node.parent_field == 'targets':
            # local variable
            self.node_types[type_variable] = ("scalar",) # this may change for different target architectures
            return SymbolRef(node.id, self.array_type) # danger of repeated declarations!
        elif self.node_types[type_variable] != ("scalar",): # sigh
            # TODO deal with modes... add a loop in case of nonsingular third dimension
            # TODO deal with slices, assumes singular 1. dimension (= 1 state/coupling variable)
            return ArrayRef(SymbolRef(node.id), SymbolRef("node_it") ) 
        else:
            return self.generic_visit(node)#SymbolRef(node.id, self.array_type())

class DfunDef(NodeTransformer):
    def __init__(self,pars):
        self.pars = pars
    def visit_Return(self,node):
        # TODO add return variable
        return None


    def visit_FunctionDef(self,node):
        node.args.args.append(
                    SymbolRef(
                    'pars',
                    ctypes.POINTER(ctypes.c_double)()
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
                    'derivative',
                    ctypes.POINTER(ctypes.c_double)()
                    )
                )
        #return node
        return self.generic_visit(node)

class LoopBody(NodeTransformer):

    def visit_FunctionDef(self,node):
        loop = [For(
                init = Assign(SymbolRef("node_it", c_int()), Constant(0)),
                test = Lt(SymbolRef("node_it"), Constant( SymbolRef('n_nodes'))),
                incr = PostInc(SymbolRef("node_it")),
                body = node.body 
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
        self.ret = None

    def write_dag(self, filename):
        for node in self.dag.nodes_iter():
            self.dag.node[node]['label'] = ast.dump(node)

        nx.write_graphml(self.dag,filename)

    def visit_FunctionDef(self,node):
        for arg in node.args.args:
            if arg.id != "self":
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

    def visit_List(self,node):
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                self.dag.add_edge(node,self.variables[elt.id])
            else:
                self.dag.add_edge(node,elt)
        self.generic_visit(node)

    def visit_Return(self,node):
        if isinstance(node.value, ast.Name):
            self.dag.add_edge(node,self.variables[node.value.id])
        else:
            self.dag.add_edge(node,node.value)
        assert(self.ret is None) # sanity check
        self.ret = node
        self.generic_visit(node)

def deps_to_types(deps, ret, sim):
    node_types = {}
    # type function params, (also self.params) ... needed?
    n_svar = len(sim.model.state_variables)
    if n_svar == 1:
        svar_d = "scalar"
    else:
        svar_d = "n_svar"

    n_modes = sim.model.number_of_modes
    if n_modes == 1:
        modes_d = "scalar"
    else:
        modes_d = "n_modes"

    n_cvar = len(sim.model.cvar)
    if n_cvar == 1:
        cvar_d = "scalar"
    else:
        cvar_d = "n_cvar"

    # TODO when implemented
    #n_lcvar = len(sim.model.lcvar)
    lcvar_d = "scalar" # TODO

    n_nodes = sim.number_of_nodes
    nodes_d = "n_nodes"

    sources = [n for n,d in deps.out_degree_iter() if d ==0]
    for s in sources:
        if isinstance(s, ast.Attribute):
            assert(s.value.id =='self')
            node_types[s] = ("scalar",) # better scalar representation?
        elif isinstance(s, ast.Name):
            assert(isinstance(s.ctx, ast.Param))
            if s.parent_field_index == 1: 
                # state variables
                node_types[s] = (svar_d, nodes_d, modes_d)
            elif s.parent_field_index == 2: 
                # coupling
                node_types[s] = (cvar_d, nodes_d, modes_d)
            elif s.parent_field_index == 3: 
                # local_coupling
                node_types[s] = (lcvar_d, nodes_d, modes_d)
            elif s.parent_field_index == 4: 
                # stimulus, TODO
                continue
            elif s.parent_field_index == 0: 
                # self, remove...
                continue
            else:
                raise Exception("dfun"+ ast.dump(s.parent) + " has too many arguments " + ast.dump(s))
        elif isinstance(s,ast.Num):
            node_types[s] = ("scalar", )
        else: 
            raise Exception("Unknown source node type " + ast.dump(s))
    arr_c = 0

    # dependence edges in resolved order (linearization from return value)
    dep_nodes = nx.topological_sort(deps)
    for node in dep_nodes:
        if node_types.has_key(node):
            continue # it is a source
        # map operators
        if isinstance(node, ast.BinOp):
            parents = [v for (u,v) in deps.out_edges([node])]
            assert(len(parents) == 2)
            if node_types[parents[0]] == node_types[parents[1]]:
                #   per element array ops
                node_types[node] = node_types[ parents[0]]
            else:
                #   per element array op scalar
                if node_types[parents[0]] == ("scalar",):
                    node_types[node] = node_types[parents[1]]
                elif node_types[parents[1]]== ("scalar",):
                    node_types[node] = node_types[parents[0]]
                else:
                    raise Exception("Binary operator dimension mismatch "+ ast.dump(node))

        if isinstance(node, ast.UnaryOp) or isinstance(node, ast.Return):
            parents = [v for (u,v) in deps.out_edges([node])]
            assert(len(parents) == 1)
            node_types[node] = node_types[ parents[0]]

        if isinstance(node, ast.List):
            parents = [v for (u,v) in deps.out_edges([node])]
            assert(len(parents) > 0)
            # this should be more generic if used outside arranging derivative for return...
            par_type = node_types[parents[0]]
            for p in parents:
                assert(node_types[p] == par_type) # more sanity checks
            node_types[node] = ("arr_" + str(arr_c),par_type[1], par_type[2])
            arr_c +=1

        if isinstance(node, ast.Call):
            # TODO be more generic than numpy functions?
            assert(isinstance(node.func, ast.Attribute) and node.func.value.id == "numpy")
            parents = [v for (u,v) in deps.out_edges([node])]
            assert(len(parents) == 1)
            node_types[node] = node_types[ parents[0]] # works both for array and per element functions

        # TODO reduce operators?
        #   slices
        if isinstance(node, ast.Subscript):
            sucs = deps.successors(node)
            assert(len(sucs)==1) # depends only on one expression
            suc = sucs[0]
            dims_source = node_types[suc]
            dims = []
            if isinstance(node.slice, ast.ExtSlice):
                for i, dim in enumerate(node.slice.dims):
                    if isinstance(dim, ast.Index):
                        dims.append("scalar")
                    elif isinstance(dim, ast.Slice):
                        assert(dim.upper is None and  dim.lower is None and dim.step is None) # very talkative... 
                        dims.append(dims_source[i])
            else:
                if isinstance(node.slice, ast.Index):
                    dims.append("scalar")
                elif isinstance(node.slice, ast.Slice):
                    assert(node.slice.upper is None and  node.slice.lower is None and node.slice.step is None) # very talkative... 
                    dims.append(dims_source[i])
                

            #nd_sub = len(node.slice.dims)
            nd_sub = len(dims)
            nd_source = len(dims_source)
            for d in range( nd_sub, nd_source):
                dims.append(dims_source[d]) 
            node_types[node] = tuple(dims)


        # assignments
        if isinstance(node, ast.Name):
            assert( isinstance(node.ctx, ast.Store) ) 
            sucs = deps.successors(node) 
            assert(len(sucs)==1) # single data dependency
            suc = sucs[0]
            node_types[node] = node_types[suc]

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

        node_types = deps_to_types(datadep.dag, datadep.ret, self.sim)

        # traverse the python AST, replace numpy slices address 
        # locate dot operations and replace accordingly
        tree = CMathConversions().visit(tree)
    
        # maybe later
        # tree = DotReplacement().visit(tree)
        params = ModelParameters()
        tree = params.visit(tree)
        pars = sorted(params.pars, key=params.pars.get)

        tree = NamesToArrays(state_vars, node_types, datadep.variables).visit(tree)
        tree = DfunDef(pars).visit(tree)
        tree = LoopBody().visit(tree)
        # not needed, for now...


        
        prog_tree = CFile("generated_dfun", 
            tree.body #dfun definition and includes
            )
        prog_tree = PyBasicConversions().visit(prog_tree)
        proj = Project([prog_tree])


        fn = CModelDfunFunction(self.sim, pars)

        dtype = self.sim.history.dtype
        typesig = CFUNCTYPE(
                None, 
                arg_config["state_vars"], 
                arg_config["coupling"],
                arg_config["local_coupling"],
                np.ctypeslib.ndpointer(
                    dtype, 
                    1,
                    shape=(len(pars))
                    ), #params
                c_int, #n_nodes
                c_int, #n_modes
                arg_config["state_vars"] # derivative
                )
        print("FUNCTYPE", typesig._restype_, typesig._argtypes_)

        return fn.finalize("dfun", proj, typesig)

