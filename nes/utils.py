"""
Functions for runtime manipulation of TVB objects. Rename to something more intuitive
"""

def specialize_model(sim):
    """
    Take out the model, feed it to the specializer and replace the dfun with specialized version. Simulation is expected to be already configured. 
    """
    py_ast = get_ast(sim.model.dfun)
    sim.model.dfun = CModelDfun(py_ast)
     

