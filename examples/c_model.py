"""
create and configure NF simulation (tvb stuff)
specialize the model (nes stuff)
...
profit (memics)
"""


import logging

logging.basicConfig(level=20)
import sys
assert len(sys.argv) == 2, "do you want specialization? [0,1]"



# TVB stochastic surface demo

from tvb.simulator.lab import *
from nes.utils import specialize_model

#Initialise a Model, Coupling, and Connectivity.
#rfhn = models.ReducedSetFitzHughNagumo()
rfhn = models.WilsonCowan()
white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=0.0043)   # 0.0066

#Initialise an Integrator
hiss = noise.Additive(nsig=numpy.array([2 ** -16, ]))
heunint = integrators.HeunStochastic(dt=2 ** -4, noise=hiss)

#Initialise some Monitors with period in physical time
mon_tavg = monitors.TemporalAverage(period=2 ** -2)
mon_savg = monitors.SpatialAverage(period=2 ** -2)

#Bundle them
what_to_watch = (mon_tavg, mon_savg)

#grey_matter = surfaces.LocalConnectivity(cutoff=40.0)
#grey_matter.equation.parameters['sigma'] = 10.0
#grey_matter.equation.parameters['amp'] = 1.0
#local_coupling_strength = numpy.array([-0.0115])


#Initialise a surface
default_cortex = surfaces.Cortex(load_default=True)
#default_cortex.local_connectivity = grey_matter
#default_cortex.coupling_strength = local_coupling_strength

#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and surface.
sim = simulator.Simulator(model=rfhn, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch,
                          surface=default_cortex)

sim.configure()

if sys.argv[1] == '1':
    specialize_model(sim)

#for _, _ in sim(simulation_length=2):
#    pass

#Perform the simulation
savg_data = []
savg_time = []

for _, savg in sim(simulation_length=2 ** 2):
    if not savg is None:
        savg_time.append(savg[0])
        savg_data.append(savg[1])


LOG.info("finished simulation")

##----------------------------------------------------------------------------##
##-               Plot pretty pictures of what we just did                   -##
##----------------------------------------------------------------------------##

##Make the lists numpy.arrays for easier use.
#SAVG = numpy.array(savg_data)
#
##Plot region averaged time series
#figure(3)
#plot(savg_time, SAVG[:, 0, :, 0])
#title("Region average")
#
#show()
