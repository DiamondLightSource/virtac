# Overview

## What is the Virtac?

The Diamond Virtual Accelerator (Virtac for short) was created to simulate the high level controls interface to the real accelerator, mostly the storage ring. There are three main parts:

1. An EPICS IOC which aims to faithfully reproduce the PVs available on the real machine and their behavior.
2. A pytac lattice which contains the same elements (dipoles, quadrupoles, etc) as the real machine and is updated via the aforementioned PV interface.
3. A pyAT simulation which is updated from the pytac lattice and indirectly from the Virtac PV interface.

### 1. PV interface

The Virtac provides over 4000 PVs which allow you to control the Virtac and to read information about it using channel access. We use [PythonSoftIOC](https://diamondlightsource.github.io/pythonSoftIOC/master/tutorials/creating-an-ioc.html) to create the EPICS IOC. We use information from pytac (which in turn derives its data from matlab middlelayer) in addition to information contained in csv files to create PVs. The data in our csv files is largely created by reading PVs from the real machine, with some manually generated configuration.

### 2. Pytac lattice

[Pytac](https://pytac.readthedocs.io/en/latest/) is used to define the lattice which describes the layout of the storage ring. The lattice tracks over 2000 elements, including different types of dipoles, quadrupoles, sextupoles, drifts and bpms.

Pytac supports many lattice modes, but the Virtac currently supports just three: 'DIAD', 'I04' and the Diamond 2 ringmode, '48'.

### 3. pyAT simulation

[pyAT](https://atcollab.github.io/at/p/index.html) is used to simulate the physics inside the particle accelerator. More specifically, we mostly use pyAT to simulate the path of a single electron around the storage ring. pyAT also contains a lattice, this is linked to and configured from the pytac lattice.

Our main use-case of pyAT is to allow modifications of the lattice (such as setting magnet currents via the PV interface) and then calculating how these modifications affect the reference orbit of an electron travelling around the storage ring. This core premise allows us to test certain feedback algorithms against the Virtac by setting magnet currents, then reading bpm positions and repeating until we find an idealized orbit.

### What can currently be tested against the Virtac?

Currently the Virtac only simulates the storage ring, but this may change in the future.

The main use-case is IOCs which write to magnet current PVs and then need to see what effect this has on the ring.

We simulate the following magnet families:
- Bends
- Quadrupoles
- Skew quadrupoles
- Horizontal and vertical steerers/correctors
- Horizontal and vertical ID trim coils
- Chicanes
- Sextupoles
- Octupoles (for Diamond 2)
- Combined dipole-quadrupoles (for Diamond 2)
- Anti-bend dipoles (for Diamond 2)

We simulate the following ring parameters and provide them as PVs:
- BPM positions
- Emittance
- Tune
 
Some of the things that we dont simulate:
- The linac
- The booster ring
- The transfer lines
- Particle injection
- Bunches
- Electrons outside of the reference orbit
- Beam decay
- Insertion devices (only the trim coils for ID corrections are simulated)

With the default settings, the simulation can only be recalculated at a maximum rate of:
- Approximately 2Hz for Diamond 1
- Approximately 0.7Hz for Diamond 2

### Apps currently being tested against the Virtac

- Slow orbit feedback
- Tune feedback
- Vertical emittance feedback
- Response matrix calculation

### Future plans

- Allow the Virtac to be run using the Diamond II lattice configuration
- Simulate the booster to storage ring transfer line
- Simulate electrons which do not follow the reference orbit, ie by giving them a non-zero initial position in the beam tube.
- Reduce the simulation recalculation delay
