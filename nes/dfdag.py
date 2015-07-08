import networkx as nx

class DFDAG:
    """
    Contains list of nodes, provides entry points for algorithms.
    """
    def __init__(self, applies, values, results=[]):
        self.applies = applies
        self.values = values
        self.results = results


    def nx_representation(self):
        dg = nx.DiGraph()
        for apply in self.applies:
            for input in apply.inputs:
                dg.add_edge(apply, input)
            dg.add_edge(apply.output, apply)

        return dg

    def linearize(self):
        return nx.topological_sort(self.nx_representation())

    def to_dot(self):
        d_str = 'digraph {\n'
        for value in self.values:
            d_str += '%d [label="%s",shape=box]\n' % (id(value), str(value))
        for apply in self.applies:
            d_str += '%d [label="%s"]\n' % (id(apply), str(apply))
            for input in apply.inputs:
                d_str += "%d -> %d\n" % (id(apply), id(input))
            d_str += "%d -> %d\n" % (id(apply.output), id(apply))
        d_str += "}\n"
        return d_str


class Node(object):
    """
    There will be two basic types: functions and values. Edges are implicit.
    Function ``depends`` returns iterator over targets of outgoing edges (for
    DAG walks) and should be implemented by all nodes in the graph.
    """
    def depends(self):
        raise NotImplementedError()


class Apply(Node):
    """
    Represents routine application. Specified by Routine type.
    """

    def __init__(self, routine, inputs, output):
        # TODO fancy checking later (routine/inputs/outpus consitency), we live
        # dangerously for now
        self.routine = routine
        self.inputs = inputs
        self.output = output
        if output is not None: 
            # like for example return
            output.source = self

    def depends(self):
       return self.inputs

    def __repr__(self):
        return "<Apply: " + str(self.routine) + ">"

class Value(Node):
    """
    Represents single value (array, slice, constant). Once created, cannot be
    changed, can be reused.
    """

    def __init__(self, type=None, source=None):
        # TODO check consitency of type vs source.routine/source_index later
        self.type = type
        self.source = source

    def depends(self):
        if self.source is None:
            return []
        else:
            return [self.source]
    
    def __repr__(self):
        return "<Value: " + str(self.type) + ">"



class Type(object):
    """
    Data value type base class
    """
    pass

class ScalarType(Type):
    pass

class Constant(ScalarType):
    def __init__(self, number):
        self.number = number

class ArrayType(Type):
    """
    Represents a reference to an array, or its part (in a sense of NumPy slice).
    """
    
    def __init__(self, data, slice=None ):
        self.data = data # memory allocation
        self.slice = slice

    @property 
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, slice):
        """
        This allows only fully described slices (for the whole ArrayData). For
        slices of views (already sliced arrays), use apply_slice().
        """
        if slice is None:
            # [':', ..., ':'] take all on all dimensions
            slice = tuple([':'] * len(self.data.shape)) 
            self._slice = slice
            self._dim_map = range(len(self.data.shape)) # identity for start
        else:
            # go get the proper slice length
            assert(len(slice) == len(self.data.shape))
            self._dim_map = []
            for i, sl in enumerate(slice):
                if(sl == ":"):
                    self._dim_map.append(i)
                else:
                    assert(isinstance(sl,int)) # just simple slices for now 
                    # nontrivial slices only on known-sized dimensions
                    assert(isinstance(self.data.shape[i], int) ) 

                    # we discard dimensions where slice is integer 
                    pass
            self._slice = slice

    def apply_slice(self, slice):
        #TODO vyuzije dim_map, vytvori novy slice a aplikuje ho
        assert( len(slice) == len(self.shape) == len(self._dim_map) )
        new_slice = list(self._slice)
        for i, dim in enumerate(self._dim_map):
            new_slice[dim] = slice[i]

        self.slice = tuple(new_slice)


    @property
    def shape(self):
        newshape = []
        for i in reversed(range(len(self.data.shape))):
            if isinstance(self.slice[i], str): 
                # we don't support complex slices for now
                assert(self.slice[i] == ":") 
                newshape.append(self.data.shape[i])
            # we discard dimensions where slice is integer 
        newshape.reverse()
        newshape = tuple(newshape)
        return newshape

    @shape.setter
    def shape(self,shape):
        # just don't
        raise RuntimeError() 


    def broadcast_with(self, other):
        # correctness of the broadcast assumed
        if isinstance(other, ScalarType):
            shape =  self.shape
        else:
            assert( isinstance(other, ArrayType))
            shape = self._broadcast_shapes(self.shape, other.shape)
        return ArrayType( data=ArrayData(shape))

    # TODO refactor this to the function above
    def _broadcast_shapes(self, shape1, shape2):
        if len(shape1) < len(shape2):
            shape = shape2
        elif len(shape1) > len(shape2):
            shape = shape1
        else:
            shape = []
            for i in reversed(range(len(shape1))):
                if isinstance(shape1[i], str):
                    shape.append(shape1[i])
                elif isinstance(shape2[i], str):
                    shape.append(shape2[i])
                # we discard dimensions where both are 1
            shape.reverse()
            shape = tuple(shape)
        return shape

    def __repr__(self):
        return "<ArrayType: shape:" + str(self.shape) + " | slice:" + str(self.slice) +" | data:" + str(self.data) +">"

class ArrayData(object):
    """
    Represents pointer to array data. The shape can be mix of known-sized and
    parametric dimensions.

    shape: list of integers (dims with known size ) and strings (parametric
    dims).
    """
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return "<ArrayData: " + str(hex(id(self))) + " " + str(self.shape) + ">"


class Routine(object):
    """
    Metadata container for funcion declaration.
    """
    pass



class BinOp(Routine):
    # note: output type determines function mapping dimension
    def __init__(self, operator, input_types, output_type):
        self.input_types = input_types
        self.output_type = output_type 
        self.operator = operator

class Broadcast(Routine):
    """
    Sets the values from the whole source array to the part of target array.
    Syntactically it represents an assign to subscripted variable:  x[0,:] = a.

    source: Value of ArrayType
    target: Value of ArrayType

    Note, that the subscript itself is stored in the target ArrayType, while
    the actual memory destination in the target ArrayData. Shape of source and
    target should to match.
    """
    def __init__(self, source, target):
        self.source = source
        self.target = target

class Synchronize(Routine):
    # array access synchronization (sliced access)
    pass

class Return(Routine):
    # represents return statement
    pass



class LoopBlock(object):
    def __init__(self, dim):
        self.dim = dim # name of loop dimension
        self.applies = set() 


