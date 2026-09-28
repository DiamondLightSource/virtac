# PVs Provided by the Virtac

Many of these PVs are, by design, identical to those used in the real machine. But there are some additional PVs either for conveniance or to simplify some behaviour.

All of these PVs can be read using ```caget``` and some PVs can be written to using ```caput```

This page only documents the PVs which are most useful to the user, a complete list of provided PVs is available by entering the ```dbl()``` command into the Virtac's interactive terminal.

## BPMs

To read the x/y position of the first bpm in the first cell:

```
caget SR01C-DI-EBPM-01:SA:X
caget SR01C-DI-EBPM-01:SA:Y
```

To read the PVs which contain an array of x/y positions from all bpms:

```
caget SR-DI-EBPM-01:SA:X
caget SR-DI-EBPM-01:SA:Y
```

## Emittance

To get the combined, x and y emittance of the ring as a whole:

```
caget SR-DI-EMIT-01:EMITTANCE
caget SR-DI-EMIT-01:HEMIT
caget SR-DI-EMIT-01:VEMIT
```

## Tune

To get the x and y tunes for the ring as a whole:

```
caget SR23C-DI-TMBF-01:TUNE:TUNE
caget SR23C-DI-TMBF-02:TUNE:TUNE
```

## Magnets

These setpoints PVs can all be read and written to using channel access. They are used to set the currents (in engineering units) to a variety of magnet families. 

Example caputs using sensible values:

```
caput SR-PC-DIPOL-01:SETI 1354.61
caput SR01A-PC-HSTR-01:SETI -0.247326
caput SR02A-PC-HSCOR-01:SETI -0.0314738
caput SR01A-PC-Q1D-01:SETI 71.545
caput SR01A-PC-S1D-01:SETI 35.3174
caput SR01A-PC-SQUAD-01:SETI 0.2689
```

To read the current value for these magnets used by the simulation:

```
caget SR-PC-DIPOL-01:I
caget SR01A-PC-HSTR-01:I
caget SR02A-PC-HSCOR-01:I
caget SR01A-PC-Q1D-01:I
caget SR01A-PC-S1D-01:I
caget SR01A-PC-SQUAD-01:I
```

For the most part the SETI and I PVs such give the same value, one exception is for the quadrupoles used by tunefeedbacks, where the SETI current is offset by the tune offset prior to being set to the I PV used in the simulation.

## Others

These PVs only store setpoints, they are not updated by the simulation, but can be used to adjust it.

The master oscillator is used to configure RF cavities:

```
caget LI-RF-MOSC-01:FREQ
caput LI-RF-MOSC-01:FREQ_SET 499687000
```

The beam current does not have any effect on the simulation, but can be read:

```
caget SR-DI-DCCT-01:SIGNAL
```
