import networkx as nx

class DFDAG:
    """
    Contains list of nodes, provides entry points for algorithms.
    """
    def __init__(self, applies, values, entries=None, results=None):
        self.applies = applies
        self.values = values

    def nx_representation(self):
        dg = nx.DiGraph()
        for apply in self.applies:
            for input in apply.inputs:
                dg.add_edge(apply, input)
            dg.add_edge(apply.output, apply)

        return dg

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
    """
    pass


class Apply(Node):
    """
    Represents routine application. Specified by Routine type.
    """

    def __init__(self, routine, inputs, output):
        # TODO fancy checking later (routine/inputs/outpus consitency), we live dangerously for now
        self.routine = routine
        self.inputs = inputs
        self.output = output
        output.source = self
    def __repr__(self):
        return "<Apply: " + str(self.routine) + ">"

class Value(Node):
    """
    Represents single value (array, slice, constant). Once created, cannot be changed, can be reused.
    """

    def __init__(self, type=None, source=None):
        # TODO check consitency of type vs source.routine/source_index later
        self.type = type
        self.source = source
    
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

    def _slice_shape(self, shape, slice):
        newshape = []
        for i in reversed(range(len(shape))):
            if isinstance(slice[i], str): #shouldnt we go for ":" notation?
                newshape.append(shape[i])
            # we discard dimensions where slice is integer
        newshape.reverse()
        newshape = tuple(newshape)
        return newshape


    @property
    def shape(self):
        if self.slice is None:
            return self.data.shape
        else:
            return self._slice_shape(self.data.shape,self.slice)

    @shape.setter
    def shape(self,shape):
        # just don't
        raise RuntimeError() 

    def __init__(self, data, slice=None ):
        self.data = data # memory allocation
        if slice is None:
            self.slice = self.data.shape
        else:
            self.slice = slice

    def broadcast_with(self, other):
        # correctness of the broadcast assumed
        if isinstance(other, ScalarType):
            shape =  self.shape
        else:
            assert( isinstance(other, ArrayType))
            shape = self._broadcast_shapes(self.shape, other.shape)
        return ArrayType( data=ArrayData(shape))

    def __repr__(self):
        return "<ArrayType: shape:" + str(self.shape) + " | slice:" + str(self.slice) +" | data:" + str(self.data) +">"

class ArrayData(object):
    """
    Represents pointer to array data.
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
    def __init__(self, input_types, output_type):
        self.input_types = input_types
        self.output_type = output_type

class Synchronize(Routine):
    # array access synchronization (sliced access)
    pass
