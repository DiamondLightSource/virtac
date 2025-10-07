"""Contains the VirtacServer class which creates and manages the PV interface to the
VIRTAC. It also provides various public methods to interact with Virtac"""

import csv
import logging
import typing
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import cast

import atip
import numpy
import pytac

from .pv import (
    BasePV,
    CollationPV,
    InversionPV,
    MonitorPV,
    ReadSimPV,
    ReadWriteSimPV,
    RecordData,
    RecordTypes,
    RecordValueType,
    RefreshPV,
    SummationPV,
)

LimitsDictType = dict[str, tuple[str, str, str, str, str, str]]


class MirrorType(StrEnum):
    BASIC = "basic"
    INVERSE = "inverse"
    SUMMATE = "summate"
    COLLATE = "collate"


MIRROR_TYPES = {
    MirrorType.BASIC: MonitorPV,
    MirrorType.INVERSE: InversionPV,
    MirrorType.SUMMATE: SummationPV,
    MirrorType.COLLATE: CollationPV,
}


class VirtacServer:
    """The soft-ioc server which contains the configuration and PVs for the VIRTAC.
    It allows ATIP to be interfaced using EPICS in the same manner as the live machine.

    Attributes:
        lattice (pytac.lattice.Lattice): An instance of a Pytac lattice with a
            simulator data source derived from pyAT.
    """

    def __init__(
        self,
        ring_mode: str,
        limits_csv: Path | None = None,
        bba_csv: Path | None = None,
        feedback_csv: Path | None = None,
        mirror_csv: Path | None = None,
        tune_csv: Path | None = None,
        linopt_function: str = "linopt6",
        disable_emittance: bool = False,
        disable_chromaticity: bool = False,
        disable_radiation: bool = False,
        disable_tunefb: bool = False,
    ) -> None:
        """
        Args:
            ring_mode: The ring mode to create the lattice in.
            limits_csv: The filepath to the .csv file from which to load the pv
                limits. For more information see create_csv.py.
            bba_csv: The filepath to the .csv file from which to load the bba
                records, for more information see create_csv.py.
            feedback_csv: The filepath to the .csv file from which to load the
                feedback records, for more information see create_csv.py.
            mirror_csv: The filepath to the .csv file from which to load the
                mirror records, for more information see create_csv.py.
            tune_csv: The filepath to the .csv file from which to load the tune
                feedback records, for more information see create_csv.py.
            disable_emittance: Whether emittance should be disabled.
            disable_tunefb: Whether tune feedback should be disabled.
        """
        self._linopt_function: str = linopt_function
        self._disable_emittance: bool = disable_emittance
        self._disable_chromaticity: bool = disable_chromaticity
        self._disable_radiation: bool = disable_radiation

        self._disable_tunefb: bool = disable_tunefb
        self._pv_monitoring: bool = True
        self.lattice: pytac.lattice.EpicsLattice = atip.utils.loader(
            ring_mode,
            self._linopt_function,
            self._disable_emittance,
            self._disable_chromaticity,
            self._disable_radiation,
            self.update_pvs,
        )
        self.lattice.set_default_data_source(pytac.SIM)
        # Holding dictionary for all PVs
        self._pv_dict: dict[str, BasePV] = {}
        # Dictionary for the PVs which should be automatically updated when the
        # simulation data is recalculated
        self._readback_pvs_dict: dict[str, ReadSimPV] = {}

        print("Starting PV creation.")
        self._create_core_pvs(limits_csv)

        if bba_csv is not None:
            self._create_bba_records(bba_csv)
        if feedback_csv is not None:
            self._create_feedback_records(feedback_csv)
        if mirror_csv is not None:
            self._create_mirror_records(mirror_csv)
        if not disable_tunefb and tune_csv is not None:
            self._setup_tune_feedback(tune_csv)

        self.print_virtac_stats()

    def update_pvs(self) -> None:
        """The callback function passed to ATSimulator during lattice creation,
        which is called each time a calculation of physics data is completed and
        updates all the in records that do not have a corresponding out record
        with the latest values from the simulator.
        """
        logging.info("Updating output PVs")
        for pv in self._readback_pvs_dict.values():
            pv.update_from_sim()
        logging.debug("Finished updating output PVs")

    def _create_core_pvs(self, limits_csv: Path | None) -> None:
        """Create the core records required for the virtac using both lattice and
        element pytac data.

        Args:
            limits_csv: The filepath to the .csv file from which to load pv field data
                to configure softioc records with.
        """
        limits_dict: LimitsDictType = {}
        if limits_csv is not None:
            try:
                with open(limits_csv) as f:
                    csv_reader = csv.DictReader(f)
                    for line in csv_reader:
                        limits_dict[line["pv"]] = (
                            str(line["upper"]),
                            str(line["lower"]),
                            str(line["precision"]),
                            str(line["drive_high"]),
                            str(line["drive_low"]),
                            str(line["scan"]),
                        )
            except FileNotFoundError:
                logging.warning(
                    f"Warning, could not find limits.csv file at {limits_csv}, limits "
                    "data will not be used."
                )

        # Create PVs from lattice elements.
        self._create_element_pvs(limits_dict)

        # Create PVs from the lattice itself.
        self._create_lattice_pvs(limits_dict)

    def _create_element_pvs(self, limits_dict: LimitsDictType) -> None:
        """Create a PV for each simulated field on each pytac lattice element.

        .. note::  One exception to the rule of one PV per field is the RF cavities.
                Currently there are 7 RF cavities and 1 harmonic RF cavity in the D2
                lattice, we set their working frequency to be that of the master
                oscillator (MOSC) frequency, meaning they all share a single PV. Really,
                the HRF cavity should be about 3x the MOSC frequency, but the HRF isnt
                simulated anyway, so it doesnt matter that we set the wrong value to it.
                In the future it should either have its own PV, or we should add a way
                to apply a scaling factor to the value we get/set.

        .. note:: For fields which have an in type record (RB) and an out type record
            (SP)we create SetpointPVs (or a derivative). SetpointPVs are used to set the
            pytac element with their SP record, the RB record merely reflects the set
            value.

        .. note:: For fields which only have an (RB) record and no (SP) record we just
            create regular PVs and we set their update_from_lattice to true. This means
            that when the Pytac lattice is updated after a PyAT physics recalculation,
            these PVs read their value from it.

        Args:
            limits_dict: A dictionary containing the limits data for
                the PVs.
        """
        rf_write_record = None
        for element in self.lattice:
            # Exception for the RF cavities, 8 of which share a single PV
            if element.type_.upper() == "RFCAVITY" and rf_write_record is not None:
                rf_write_record.append_pytac_item(element)
            else:
                for field in cast(
                    dict[str, list[str]], element.get_fields()[pytac.SIM]
                ):
                    try:
                        value = element.get_value(
                            field, units=pytac.ENG, data_source=pytac.SIM
                        )
                    except pytac.exceptions.UnitsException as e:
                        # For D2 the conversion is not currently working as AP has not
                        # yet updated their 'calibration_Data' matlab structure with
                        # realistic values. This means that when we attempt conversion
                        # we get halfway through, and then have the wrong values and
                        # the converted value becomes too large for the limits.
                        value = 0
                        print(e)

                    read_pv_name = cast(str, element.get_pv_name(field, pytac.RB))

                    if read_pv_name in self._pv_dict.keys():
                        print(f"PV: {read_pv_name} already exists! Dupe!")
                        continue

                    upper, lower, precision, drive_high, drive_low, scan = (
                        limits_dict.get(
                            read_pv_name, (None, None, None, None, None, "I/O Intr")
                        )
                    )
                    record_data = RecordData(
                        RecordTypes.AI,
                        lower=lower,
                        upper=upper,
                        precision=precision,
                        drive_high=drive_high,
                        drive_low=drive_low,
                        initial_value=value,
                        scan=scan,
                    )

                    read_pv = ReadSimPV(
                        read_pv_name, record_data, pytac_items=[element], field=field
                    )
                    self._pv_dict[read_pv_name] = read_pv

                    # Readback PVs without a setpoint PV are updated from the simulation
                    # after recalculation. Readback PVs with a setpoint PV are updated
                    # when their associated setpoint PV is updated.
                    try:
                        read_write_pv_name = cast(
                            str, element.get_pv_name(field, pytac.SP)
                        )
                    except pytac.exceptions.HandleException:
                        # Only triggered if this element has an RB PV but no SP PV.
                        # Add to list of PVs to be updated from the simulation
                        self._readback_pvs_dict[read_pv_name] = read_pv
                    else:
                        upper, lower, precision, drive_high, drive_low, scan = (
                            limits_dict.get(
                                read_write_pv_name,
                                (None, None, None, None, None, "Passive"),
                            )
                        )
                        record_data = RecordData(
                            RecordTypes.AO,
                            lower=lower,
                            upper=upper,
                            precision=precision,
                            drive_high=drive_high,
                            drive_low=drive_low,
                            initial_value=value,
                            always_update=True,
                        )
                        read_write_pv = ReadWriteSimPV(
                            read_write_pv_name,
                            record_data,
                            read_pv,
                            pytac_items=[element],
                            pytac_field=field,
                        )
                        self._pv_dict[read_write_pv_name] = read_write_pv

                        if element.type_.upper() == "RFCAVITY":
                            rf_write_record = read_write_pv

    def _create_lattice_pvs(self, limits_dict: LimitsDictType) -> None:
        """Create a PV for each simulated field on each pytac lattice itself.

        .. note:: For fields which have an in type record (RB) and an out type record
            (SP) we create SetpointPVs (or a derivative). SetpointPVs are used to set
            the pytac element with their SP record, the RB record merely reflects the
            set value.

        .. note:: For fields which only have an (RB) record and no (SP) record we just
            create regular PVs and we set their update_from_lattice to true. This means
            that when the pytac lattice is recalculated, these PVs read their value from
            the lattice.

        Args:
            limits_dict: A dictionary containing the limits data for the PVs.
        """
        lat_field_dict = cast(dict[str, list[str]], self.lattice.get_fields())
        lat_field_set = set(lat_field_dict[pytac.LIVE]) & set(lat_field_dict[pytac.SIM])
        if self._disable_emittance:
            lat_field_set -= {"emittance_x", "emittance_y"}
        for field in lat_field_set:
            # Ignore basic devices as they do not have PVs.
            if not isinstance(
                self.lattice.get_device(field), pytac.device.SimpleDevice
            ):
                get_pv_name = cast(str, self.lattice.get_pv_name(field, pytac.RB))
                upper, lower, precision, _, _, scan = limits_dict.get(
                    get_pv_name, (None, None, None, None, None, "I/O Intr")
                )
                value = self.lattice.get_value(
                    field, units=pytac.ENG, data_source=pytac.SIM
                )
                record_data = RecordData(
                    RecordTypes.AI,
                    lower=lower,
                    upper=upper,
                    precision=precision,
                    scan=scan,
                    initial_value=value,
                )
                read_pv = ReadSimPV(
                    get_pv_name, record_data, pytac_items=[self.lattice], field=field
                )
                self._pv_dict[get_pv_name] = read_pv
                self._readback_pvs_dict[get_pv_name] = read_pv

    def _create_bba_records(self, bba_csv: Path) -> None:
        """Create all the beam-based-alignment records from the .csv file at the
        location passed, see create_csv.py for more information.

        Args:
            bba_csv: The filepath to the .csv file to load the records in accordance
                with.
        """
        self._create_feedback_or_bba_records_from_csv(bba_csv)

    def _create_feedback_records(self, feedback_csv: Path) -> None:
        """Create all the feedback records from the .csv file at the location
        passed, see create_csv.py for more information; records for one edge
        case are also created.

        Args:
            feedback_csv: The filepath to the .csv file to load the records in
                accordance with.
        """
        self._create_feedback_or_bba_records_from_csv(feedback_csv)

        # We can choose to not calculate emittance as it is not always required,
        # which decreases computation time.
        if not self._disable_emittance:
            name = "SR-DI-EMIT-01:STATUS"
            record_data = RecordData(RecordTypes.MBBI, zrvl="0", zrst="Successful")
            emit_status_pv = BasePV(name, record_data)
            self._pv_dict[name] = emit_status_pv

    def _create_feedback_or_bba_records_from_csv(self, csv_file: Path) -> None:
        """Read the csv file and create the corresponding records based on
        its contents.

        Args:
            csv_file: The filepath to the .csv file to load the records in accordance
                with.
        """
        # We don't set limits or precision but this shouldn't be an issue as these
        # records aren't intended to be set to by a user.
        with open(csv_file) as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                val: typing.Any = 0
                name = line["pv"]
                try:
                    # Waveform records may have values stored as a list such as: [5 1 3]
                    # We convert that into a numpy array for initialising the record
                    if (line["value"][0], line["value"][-1]) == ("[", "]"):
                        val = numpy.fromstring((line["value"])[1:-1], sep=" ")
                    else:
                        val = float(line["value"])
                except (AssertionError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid initial value for {line['record_type']} record: "
                        f"{line['value']}"
                    ) from exc
                else:
                    record_data = RecordData(line["record_type"], initial_value=val)
                    pv = ReadSimPV(
                        name,
                        record_data,
                        [self.lattice[int(line["index"]) - 1]],
                        line["field"],
                    )
                    self._pv_dict[name] = pv

    def _create_mirror_records(self, mirror_csv: Path) -> None:
        """Create all the mirror records from the .csv file at the location
        passed, see create_csv.py for more information.

        Args:
            mirror_csv : The filepath to the .csv file to load the records in accordance
                with.
        """
        with open(mirror_csv) as f:
            val: RecordValueType = 0
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                # Get a list of input pvs, these are all virtac owned pvs
                input_pv_names = line["in_pv"].split(", ")
                if (len(input_pv_names) > 1) and (line["mirror_type"] in ["basic"]):
                    raise IndexError(
                        "Transformation mirror type takes only one input PV."
                    )
                elif (len(input_pv_names) < 2) and (
                    line["mirror_type"] in ["collate", "summate"]
                ):
                    raise IndexError(
                        "collation and summation mirror types take at least two input "
                        "PVs."
                    )
                input_records: list[BasePV] = []
                for pv in input_pv_names:
                    try:
                        # Lookup pv in our dictionary of softioc records
                        input_records.append(self._pv_dict[pv])
                    except KeyError:
                        logging.exception(f"PV {pv} does not exist within virtac")

                try:
                    # Waveform records may have values stored as a list such as: [5 1 3]
                    # We convert that into a numpy array for initialising the record
                    if (line["value"][0], line["value"][-1]) == ("[", "]"):
                        val = numpy.fromstring((line["value"])[1:-1], sep=" ")
                    else:
                        val = float(line["value"])

                except (AssertionError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid initial value for {line['output_type']} record: "
                        f"{line['value']}"
                    ) from exc

                else:
                    out_pv_name = line["out_pv"]
                    record_data = RecordData(
                        line["output_type"],
                        initial_value=val,
                        scan=line["scan"],
                    )
                    try:
                        mirror_type = MIRROR_TYPES[MirrorType(line["mirror_type"])]
                        if mirror_type == MIRROR_TYPES[MirrorType.BASIC]:
                            mirror_type = cast(type[MonitorPV], mirror_type)
                            # MonitorPV requires a list of str rather than a list of PV
                            output_pv = mirror_type(
                                out_pv_name, record_data, input_pv_names
                            )
                        else:
                            mirror_type = cast(
                                type[InversionPV]
                                | type[SummationPV]
                                | type[CollationPV],
                                mirror_type,
                            )
                            output_pv = mirror_type(
                                out_pv_name, record_data, input_records
                            )
                    except KeyError as e:
                        raise TypeError(
                            f"{line['mirror_type']} is not valid, please use one of: "
                            f"({', '.join(n.value for n in MirrorType)})"
                        ) from e

                    self._pv_dict[out_pv_name] = output_pv

    def _setup_tune_feedback(self, tune_csv: Path) -> None:
        """Read the tune feedback .csv and find the associated offset PVs,
        before starting monitoring them for a change to mimic the behaviour of
        the quadrupoles used by the tune feedback system on the live machine.

        .. note:: This is intended to be on the recieving end of the tune
           feedback system and doesn't actually perfom tune feedback itself.

        .. note:: The 'offset_pv' is the PV which monitors a 'delta_pv' and when the
            'delta_pv' changes, stores its value and triggers the 'set_pv' to process.
            When the 'set_pv' processes, it gets the value we just stored to the
            'offset_pv' and adds it to its own value.

        Args:
            tune_csv: A path to a tune feedback .csv file to be used instead of the
                default filepath passed at startup.
        """
        with open(tune_csv) as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                assert isinstance(self._pv_dict[line["set_pv"]], ReadWriteSimPV)

                self._pv_dict[line["offset_pv"]]
                set_record = cast(ReadWriteSimPV, self._pv_dict[line["set_pv"]])
                old_set_record = cast(ReadWriteSimPV, self._pv_dict[line["offset_pv"]])

                # We overwrite the old_set_record with the new RefreshPV which has
                # the required capabilities for tunefb
                new_set_record = RefreshPV(
                    line["offset_pv"], line["delta_pv"], set_record, old_set_record
                )
                set_record.attach_offset_record(new_set_record)
                self._pv_dict[line["offset_pv"]] = new_set_record

    def enable_monitoring(self) -> None:
        """Enable monitoring for all MonitorPV derived PVs. This will allow
        tune feedback and vertical emittance feedback to work again
        """
        if self._pv_monitoring:
            logging.warning("PV monitoring is already enabled, nothing to do.")
        else:
            logging.info("Enabling PV monitoring")
            for pv in self._pv_dict.values():
                if isinstance(pv, MonitorPV):
                    pv.enable_monitoring()
            self._pv_monitoring = True

    # TODO: Is this needed? It essentially just pauses a subset of the virtacs
    # functionality
    def disable_monitoring(self) -> None:
        """Disable monitoring for all MonitorPV derived PVs. This will disable
        tune feedback and vertical emittance feedback
        """
        if not self._pv_monitoring:
            logging.warning("PV monitoring is already disabled, nothing to do.")
        else:
            logging.info("Disabling PV monitoring")
            for _, pv in self._pv_dict.items():
                if isinstance(pv, MonitorPV):
                    pv.disable_monitoring()
            self._pv_monitoring = False

    def print_virtac_stats(self, verbosity: int = 0) -> None:
        """Print helpful statistics based on passed verbosity level

        Args:
            verbosity: The verbosity level to print at, higher levels print more
                information.
        """
        pv_type_count: dict[type[BasePV], int] = defaultdict(int)

        for pv in self._pv_dict.values():
            pv_type_count[type(pv)] += 1

        print("Virtac stats:")
        print(
            "\t Tune feedbacks is "
            f"{('disabled' if self._disable_tunefb else 'enabled')}"
        )
        print(f"\t Linear optics function is {self._linopt_function}")
        print(
            "\t Emittance calculations are "
            f"{('disabled' if self._disable_emittance else 'enabled')}"
        )
        print(
            "\t Chromaticity calculations are "
            f"{('disabled' if self._disable_chromaticity else 'enabled')}"
        )
        print(
            "\t Radiation calculations are "
            f"{('disabled' if self._disable_chromaticity else 'enabled')}"
        )
        print(
            f"\t PV monitoring is {('enabled' if self._pv_monitoring else 'disabled')}"
        )

        print(f"\t Total pvs: {len(self._pv_dict)}, consisting of:")
        for pv_type, count in pv_type_count.items():
            print(f"\t\t {pv_type.__name__} pvs: {count}")
        print(
            "\t Number of PVs to update after simulation recalculation: "
            f"{len(self._readback_pvs_dict)}"
        )

        if verbosity >= 1:
            print("\t Available PVs")
            for pv in self._pv_dict.values():
                print(f"\t\t {pv.name}, {type(pv)}")
