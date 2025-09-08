# Overview

## What is Virtac?

The Diamond Virtual Accelerator (virtac for short) was created to simulate the high level
controls interface to the real accelerator, mostly the storage ring. There are three main parts:

1. An EPICS IOC which aims to faithfully reproduce the PVs available on the real machine and their behavior.
2. A pytac lattice which contains the same elements (dipoles, quadrupoles, etc) as the real machine and is updated via the aforementioned PV interface.
3. A pyAT simulation which is updated from the pytac lattice and indirectly from the virtac PV interface.

### 1. PV interface

Virtac provides over 4000 PVs which allow you to control the virtac and to read information about it using channel access. We use [PythonSoftIOC](https://diamondlightsource.github.io/pythonSoftIOC/master/tutorials/creating-an-ioc.html) to create the EPICS IOC. We use information from pytac (which in turn derives its data from matlab middlelayer) in addition to information contained in csv files to create PVs. The data in our csv files is largely created by reading PVs from the real machine, with some manually generated configuration.

### 2. Pytac lattice

[Pytac](https://pytac.readthedocs.io/en/latest/) is used to define the lattice which describes the layout of the storage ring. The lattice tracks over 2000 elements, including different types of dipoles, quadrupoles, sextupoles, drifts and bpms.

We currently support two lattice modes: DIAD and I04

### 3. PyAt simulation

(pyAT)[https://atcollab.github.io/at/p/index.html] is used to simulate the physics inside the particle accelerator. More specifically, we mostly use pyAT to simulate the path of a single electron around the storage ring. pyAT also contains a lattice, this is linked to and configured from the pytac lattice.

Our main use-case of pyAT is to allow modifications of the lattice (such as setting magnet currents via the PV interface) and then calculating how these modifications affect the path travelled by an "average" electron. This core premise allows us to test our feedback algorithms against the Virtac by setting magnet currents, then reading bpm positions and repeating until we find an idealized orbit.

### What can currently be tested against Virtac?

Any tool which involves setting magnet currents and reading bpm positions

### Apps currently being tested against Virtac

- Slow orbit feedback
- Tune feedback
- Vertical emittance feedback
- Response matrix calculation

### Future plans

- Allow the Virtac to be run using the Diamond II lattice configuration
- Simulate the transfer line
