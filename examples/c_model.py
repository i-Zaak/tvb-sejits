"""
create and configure NF simulation (tvb stuff)
specialize the model (nes stuff)
...
profit (memics)
"""

# TVB stochastic surface demo

from tvb.simulator.lab import *
import nes

#Initialise a Model, Coupling, and Connectivity.
rfhn = models.ReducedSetFitzHughNagumo()
white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=0.0043)   # 0.0066

#Initialise an Integrator
hiss = noise.Additive(nsig=numpy.array([2 ** -16, ]))
heunint = integrators.HeunStochastic(dt=2 ** -4, noise=hiss)

#Initialise some Monitors with period in physical time
mon_tavg = monitors.TemporalAverage(period=2 ** -2)
mon_savg = monitors.SpatialAverage(period=2 ** -2)
mon_eeg = monitors.EEG(period=2 ** -2)

#Bundle them
what_to_watch = (mon_tavg, mon_savg, mon_eeg)

#Initialise a surface
default_cortex = surfaces.Cortex.from_file()
default_cortex.local_connectivity = surfaces.LocalConnectivity(load_default=True)

#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and surface.
sim = simulator.Simulator(model=rfhn, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch,
                          surface=default_cortex)

sim.configure()

nes.specialize_model(sim)

#execute, compare...
