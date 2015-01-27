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

class Value(Node):
    """
    Represents single value (array, slice, constant). Once created, cannot be changed, can be reused.
    """

    def __init__(self, type=None, source=None):
        # TODO check consitency of type vs source.routine/source_index later
        self.type = type
        self.source = source


class Type(object):
    """
    Data value type base class
    """
    pass

class ScalarType(Type):
    pass

class ArrayType(Type):
    def __init__(self,shape, slice=None):
        self.shape = shape
        if slice is None:
            self.slice = shape
        else:
            self.slice = slice




class Routine(object):
    """
    Metadata container for funcion declaration.
    """

    def __init__(self, input_types, output_type):
        self.input_types = input_types
        self.output_type = output_type


class ElemwiseBinOp(Routine):
    pass

class BinOp(Routine):
    pass
