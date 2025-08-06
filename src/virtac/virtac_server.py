import csv
import logging
import typing

import atip
import numpy
import pytac
from pytac.device import SimpleDevice
from pytac.exceptions import FieldException, HandleException

from .pv import (
    PV,
    CaPV,
    CollationPV,
    InversionPV,
    MonitorPV,
    OffsetPV,
    PVType,
    ReadbackPV,
    RecordData,
    RecordValueType,
    RefreshPV,
    SetpointPV,
    SummationPV,
)


class VirtacServer:
    """The soft-ioc server which contains the configuration and PVs for the VIRTAC.
    It allows ATIP to be interfaced using EPICS, in the same manner as the live machine.

    Args:
        ring_mode (str): The ring mode to create the lattice in.
        limits_csv (str): The filepath to the .csv file from which to
                                load the pv limits, for more information
                                see create_csv.py.
        bba_csv (str): The filepath to the .csv file from which to
                                load the bba records, for more
                                information see create_csv.py.
        feedback_csv (str): The filepath to the .csv file from which to
                                load the feedback records, for more
                                information see create_csv.py.
        mirror_csv (str): The filepath to the .csv file from which to
                                load the mirror records, for more information
                                see create_csv.py.
        tune_csv (str): The filepath to the .csv file from which to
                            load the tune feedback records, for more
                            information see create_csv.py.
        _disable_emittance (bool): Whether emittance should be disabled.
        _disable_tunefb (bool): Whether tune feedback should be disabled.

    Attributes:
        lattice (pytac.lattice.Lattice): An instance of a Pytac lattice with a
                                          simulator data source.
        tune_feedback_enabled (bool): A boolean indicating whether the tune
                                      feedback records have been created and
                                      the monitoring systems are running.
        _pv_monitoring (bool): Whether the mirrored PVs are being monitored.
        _tune_fb_csv_path (str): The path to the tune feedback .csv file.
        _pv_dict (dict[PVType]): A dictionary containing every PV created by the
            virtac with the PV name as the key and PV object as the item in a 1 to 1
            mapping.
        _readback_pvs_dict (dict[PVType]): A dictionary containing the subset of
            pvs from _pv_dict which need updating whenever the pytac lattice changes.
    """

    def __init__(
        self,
        ring_mode: str,
        limits_csv: str,
        bba_csv: str | None = None,
        feedback_csv: str | None = None,
        mirror_csv: str | None = None,
        tune_csv: str | None = None,
        disable_emittance: bool = False,
        disable_tunefb: bool = False,
    ):
        self._disable_emittance: bool = disable_emittance
        self._disable_tunefb: bool = disable_tunefb
        self._pv_monitoring: bool = True
        # TODO: Need to update ATIP to use enable_emittance instead of disable_emittance
        self.lattice: pytac.lattice.EpicsLattice = atip.utils.loader(
            ring_mode, self.update_pvs, self._disable_emittance
        )

        self._pv_dict: dict[str, PVType] = {}
        self._readback_pvs_dict: dict[str, PV] = {}
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

        # Collect the PVs that need updating after lattice recalculation.
        for name, pv in self._pv_dict.items():
            if isinstance(pv, ReadbackPV):
                self._readback_pvs_dict[name] = pv

        self.print_virtac_stats()

    # TODO: Reset all correctors to default method?

    def update_pvs(self):
        """The callback function passed to ATSimulator during lattice creation,
        which is called each time a calculation of physics data is completed and
        updates all the in records that do not have a corresponding out record
        with the latest values from the simulator.

            - Note that a PV can have multiple elements, specifically for the bend
              magnets. Currently we just have 1 PV for all bends and it takes its
              value from element[0]. This could be a target for future improvement.
        """
        logging.debug("Updating output PVs")
        for name, pv in self._readback_pvs_dict.items():
            logging.debug(f"Updating pv {name}")
            elements, field = pv.get_pytac_data()
            try:
                value = elements[0].get_value(
                    field, units=pytac.ENG, data_source=pytac.SIM
                )
                logging.debug(f"Update_pvs: {name} to val {value}")
                pv.set(value)
            except FieldException as e:
                print("PV is missing an expected pytac field")
                raise (e)
        logging.debug("Finished updating output PVs")

    def _create_core_pvs(self, limits_csv: str):
        """Create the core records required for the virtac from both lattice and element
        pytac fields. Several assumptions have been made for simplicity and
        efficiency, these are:

        Args:
            limits_csv (str): The filepath to the .csv file from which to load pv field
                              data to configure softioc records with.
        """
        limits_dict: dict = {}
        if limits_csv is not None:
            with open(limits_csv) as f:
                csv_reader = csv.DictReader(f)
                for line in csv_reader:
                    limits_dict[line["pv"]] = (
                        float(line["upper"]),
                        float(line["lower"]),
                        int(line["precision"]),
                        float(line["drive_high"]),
                        float(line["drive_low"]),
                        str(line["scan"]),
                    )

        # Create PVs from lattice elements.
        self._create_element_pvs(limits_dict)

        # Create PVs from the lattice itself.
        self._create_lattice_pvs(limits_dict)

    def _create_element_pvs(self, limits_dict: dict):
        """Create a PV for each simulated field on each pytac lattice element.

            .. Note:: The one exception to the rule of one PV per field is for the bend
                      magnets. Each of the 50 bend magnets shares the same PV and has
                      the same current value.
            .. Note:: For fields which have an in type record (RB) and an out type
                      record (SP) we create SetpointPVs (or a derivative). SetpointPVs
                      are used to set the pytac element with their SP record, the RB
                      record merely reflects the set value.
            .. Note:: For fields which only have an (RB) record and no (SP) record we
                      just create regular PVs and we set their update_from_lattice to
                      true. This means that when the pytac lattice is recalculated,
                      these PVs read their value from the lattice.

        Args:
            limits_csv (str): The filepath to the .csv file from which to load pv field
                              data to configure softioc records with.
        """
        bend_in_record = None
        for element in self.lattice:
            # There is only 1 bend PV for all bend magnets, each bend element is added
            # to this PV
            if element.type_.upper() == "BEND" and bend_in_record is not None:
                bend_in_record.append_pytac_item(element)
            else:
                for field in element.get_fields()[pytac.SIM]:
                    readback_only_pv = False
                    value = element.get_value(
                        field, units=pytac.ENG, data_source=pytac.SIM
                    )
                    read_pv_name = element.get_pv_name(field, pytac.RB)
                    try:
                        set_pv_name = element.get_pv_name(field, pytac.SP)
                    except HandleException:
                        # Only update the pv when the pytac lattice is recalculated
                        # if the RB has no corresponding SP
                        readback_only_pv = True

                    upper, lower, precision, drive_high, drive_low, scan = (
                        limits_dict.get(
                            read_pv_name, (None, None, None, None, None, "I/O Intr")
                        )
                    )
                    record_data = RecordData(
                        "ai",
                        lower=lower,
                        upper=upper,
                        precision=precision,
                        drive_high=drive_high,
                        drive_low=drive_low,
                        initial_value=value,
                        scan=scan,
                    )
                    if readback_only_pv:
                        read_pv = ReadbackPV(read_pv_name, record_data)  # type: ignore[assignment]
                    else:
                        read_pv = PV(read_pv_name, record_data)  # type: ignore[assignment]
                    read_pv.append_pytac_item(element)
                    read_pv.set_pytac_field(field)
                    self._pv_dict[read_pv_name] = read_pv

                    if not readback_only_pv:
                        upper, lower, precision, drive_high, drive_low, scan = (
                            limits_dict.get(
                                set_pv_name, (None, None, None, None, None, "I/O Intr")
                            )
                        )
                        record_data = RecordData(
                            "ao",
                            lower=lower,
                            upper=upper,
                            precision=precision,
                            drive_high=drive_high,
                            drive_low=drive_low,
                            initial_value=value,
                            always_update=True,
                        )

                        # For tunefb the quadrapole SETI records need to be OffsetPVs
                        # instead of SetpointPVs
                        if not self._disable_tunefb and field == "b1":
                            set_pv = OffsetPV(set_pv_name, record_data, read_pv)  # type: ignore[assignment]
                        else:
                            set_pv = SetpointPV(set_pv_name, record_data, read_pv)  # type: ignore[assignment]

                        self._pv_dict[set_pv_name] = set_pv
                        if element.type_.upper() == "BEND" and bend_in_record is None:
                            bend_in_record = read_pv

    def _create_lattice_pvs(self, limits_dict: dict):
        """Create a PV for each simulated field on each pytac lattice element.

            .. Note:: The one exception to the rule of one PV per field is for the bend
                      magnets. Each of the 50 bend magnets shares the same PV and has
                      the same current value.
            .. Note:: For fields which have an in type record (RB) and an out type
                      record (SP) we create SetpointPVs (or a derivative). SetpointPVs
                      are used to set the pytac element with their SP record, the RB
                      record merely reflects the set value.
            .. Note:: For fields which only have an (RB) record and no (SP) record we
                      just create regular PVs and we set their update_from_lattice to
                      true. This means that when the pytac lattice is recalculated,
                      these PVs read their value from the lattice.

        Args:
            limits_csv (str): The filepath to the .csv file from which to load pv field
                              data to configure softioc records with.
        """
        lat_fields = self.lattice.get_fields()
        lat_fields = set(lat_fields[pytac.LIVE]) & set(lat_fields[pytac.SIM])
        if self._disable_emittance:
            lat_fields -= {"emittance_x", "emittance_y"}
        for field in lat_fields:
            # Ignore basic devices as they do not have PVs.
            if not isinstance(self.lattice.get_device(field), SimpleDevice):
                get_pv_name = self.lattice.get_pv_name(field, pytac.RB)
                upper, lower, precision, drive_high, drive_low, scan = limits_dict.get(
                    get_pv_name, (None, None, None, None, None, "I/O Intr")
                )
                value = self.lattice.get_value(
                    field, units=pytac.ENG, data_source=pytac.SIM
                )
                record_data = RecordData(
                    "ai",
                    lower=lower,
                    upper=upper,
                    precision=precision,
                    scan=scan,
                    initial_value=value,
                )
                in_pv = ReadbackPV(get_pv_name, record_data)
                in_pv.append_pytac_item(self.lattice)
                in_pv.set_pytac_field(field)
                self._pv_dict[get_pv_name] = in_pv

    def _create_bba_records(self, bba_csv: str):
        """Create all the beam-based-alignment records from the .csv file at the
        location passed, see create_csv.py for more information.

        Args:
            bba_csv (str): The filepath to the .csv file to load the
                                    records in accordance with.
        """
        self._create_feedback_or_bba_records_from_csv(bba_csv)

    def _create_feedback_records(self, feedback_csv: str):
        """Create all the feedback records from the .csv file at the location
        passed, see create_csv.py for more information; records for one edge
        case are also created.

        Args:
            feedback_csv (str): The filepath to the .csv file to load the
                                    records in accordance with.
        """
        self._create_feedback_or_bba_records_from_csv(feedback_csv)

        # We can choose to not calculate emittance as it is not always required,
        # which decreases computation time.
        if not self._disable_emittance:
            name = "SR-DI-EMIT-01:STATUS"
            record_data = RecordData(
                "mbbi",
                zrvl="0",
                zrst="Successful",
            )
            emit_status_pv = PV(name, record_data)
            self._pv_dict[name] = emit_status_pv

    def _create_feedback_or_bba_records_from_csv(self, csv_file: str):
        """Read the csv file and create the corresponding records based on
        its contents.

        Args:
            csv_file (str): The filepath to the .csv file to load the
                                    records in accordance with.
        """
        # We don't set limits or precision but this shouldn't be an issue as these
        # records aren't really intended to be set to by a user.
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
                    pv = PV(name, record_data)
                    pv.set_pytac_field(line["field"])
                    pv.append_pytac_item(self.lattice[int(line["index"]) - 1])
                    self._pv_dict[name] = pv

    def _create_mirror_records(self, mirror_csv: str):
        """Create all the mirror records from the .csv file at the location
        passed, see create_csv.py for more information.

        Args:
            mirror_csv (str): The filepath to the .csv file to load the
                                    records in accordance with.
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
                # Convert input pvs to record objects
                input_records: list[PVType] = []
                for pv in input_pv_names:
                    try:
                        # Lookup pv in our dictionary of softioc records
                        input_records.append(self._pv_dict[pv])
                    except KeyError:
                        # If not owned by us, then we get it from CA
                        input_records.append(CaPV(pv))
                # Update the mirror dictionary.
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
                    if line["mirror_type"] == "basic":
                        output_pv = MonitorPV(
                            out_pv_name, record_data, [pv.name for pv in input_records]
                        )
                    elif line["mirror_type"] == "inverse":
                        output_pv = InversionPV(out_pv_name, record_data, input_records)
                    elif line["mirror_type"] == "summate":
                        output_pv = SummationPV(out_pv_name, record_data, input_records)
                    elif line["mirror_type"] == "collate":
                        output_pv = CollationPV(out_pv_name, record_data, input_records)
                    else:
                        raise TypeError(
                            f"{line['mirror type']} is not a valid mirror type; please "
                            "enter a currently supported type from: 'basic', 'summate',"
                            " 'collate' and 'inverse'."
                        )

                    self._pv_dict[out_pv_name] = output_pv

    def _setup_tune_feedback(self, tune_csv: str):
        """Read the tune feedback .csv and find the associated offset PVs,
        before starting monitoring them for a change to mimic the behaviour of
        the quadrupoles used by the tune feedback system on the live machine.

        .. Note:: This is intended to be on the recieving end of the tune
           feedback system and doesn't actually perfom tune feedback itself.

        Args:
            tune_csv (str): A path to a tune feedback .csv file to be used
                             instead of the default filepath passed at startup.
        """
        if tune_csv is None:
            raise ValueError(
                "No tune feedback .csv file was given at "
                "start-up, please provide one now; i.e. "
                "server.start_tune_feedback('<path_to_csv>')"
            )
        with open(tune_csv) as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                assert isinstance(
                    self._pv_dict[line["set_pv"]], OffsetPV
                )  # The PV which does the offsetting
                self._pv_dict[line["offset_pv"]]  # The PV which stores the offset value
                set_record: OffsetPV = self._pv_dict[line["set_pv"]]  # type: ignore[assignment]
                old_offset_record: OffsetPV = self._pv_dict[line["offset_pv"]]  # type: ignore[assignment]

                # We overwrite the old_offset_record with the new RefreshPV which has
                # the required capabilities for tunefb
                new_offset_record = RefreshPV(
                    line["offset_pv"],
                    line["delta_pv"],
                    set_record,
                    old_offset_record,
                )
                set_record.attach_offset_record(new_offset_record)
                self._pv_dict[line["offset_pv"]] = new_offset_record

    # TODO: Needs fixing from refactor, is this function needed?
    def set_feedback_record(self, index, field, value):
        """Set a value to the feedback in records.

        possible element fields are:
            ['error_sum', 'enabled', 'state', 'offset', 'golden_offset', 'bcd_offset',
             'bba_offset']
        possible lattice fields are:
            ['beam_current', 'feedback_status', 'bpm_id', 'emittance_status',
             'fofb_status', 'cell_<cell_number>_excite_start_times',
             'cell_<cell_number>_excite_amps', 'cell_<cell_number>_excite_deltas',
             'cell_<cell_number>_excite_ticks', 'cell_<cell_number>_excite_prime']

        Args:
            index (int): The index of the element on which to set the value;
                          starting from 1, 0 is used to set on the lattice.
            field (str): The field to set the value to.
            value (number): The value to be set.

        Raises:
            pytac.FieldException: If the lattice or element does
                                              not have the specified field.
        """
        try:
            self._feedback_records[(index, field)].set(value)
            self._bba_records[(index, field)].set(value)
        except KeyError as exc:
            if index == 0:
                raise FieldException(
                    f"Simulated lattice {self.lattice} does not have field {field}."
                ) from exc
            else:
                raise FieldException(
                    f"Simulated element {self.lattice[index]} does not have "
                    f"field {field}."
                ) from exc

    # Is this needed? It essentially just pauses a subset of the virtacs functionality
    def disable_monitoring(self):
        """Disable monitoring for all MonitorPV derived PVs. This will disable
        tune feedback and vertical emittance feedback
        """
        if not self._pv_monitoring:
            logging.warning("PV monitoring is already disabled, nothing to do.")
        else:
            logging.info("Disabling PV monitoring")
            for _, pv in self._pv_dict.items():
                if isinstance(pv, MonitorPV) or issubclass(type(pv), MonitorPV):
                    pv.toggle_monitoring(False)
            self._pv_monitoring = False

    def enable_monitoring(self):
        """Enable monitoring for all MonitorPV derived PVs. This will allow
        tune feedback and vertical emittance feedback to work again
        """
        if self._pv_monitoring:
            logging.warning("PV monitoring is already enabled, nothing to do.")
        else:
            logging.info("Enabling PV monitoring")
            for pv in self._pv_dict.values():
                if isinstance(pv, MonitorPV) or issubclass(type(pv), MonitorPV):
                    pv.toggle_monitoring(True)
            self._pv_monitoring = True

    def print_virtac_stats(self, verbosity: int = 0):
        """Print helpful statistics based on passed verbosity level"""
        num_pvs_dict = dict.fromkeys(
            [
                "num_ca_pvs",
                "num_pvs",
                "num_readback_pvs",
                "num_direct_pvs",
                "num_offset_pvs",
                "num_monitor_pvs",
                "num_collation_pvs",
                "num_inverse_pvs",
                "num_summation_pvs",
                "num_refresh_pvs",
            ],
            0,
        )
        total_num_pvs = len(self._pv_dict)
        for pv in self._pv_dict.values():
            if type(pv) is CaPV:
                num_pvs_dict["num_ca_pvs"] += 1
            elif type(pv) is PV:
                num_pvs_dict["num_pvs"] += 1
            elif type(pv) is ReadbackPV:
                num_pvs_dict["num_readback_pvs"] += 1
            elif type(pv) is SetpointPV:
                num_pvs_dict["num_direct_pvs"] += 1
            elif type(pv) is OffsetPV:
                num_pvs_dict["num_offset_pvs"] += 1
            elif type(pv) is MonitorPV:
                num_pvs_dict["num_monitor_pvs"] += 1
            elif type(pv) is CollationPV:
                num_pvs_dict["num_collation_pvs"] += 1
            elif type(pv) is InversionPV:
                num_pvs_dict["num_inverse_pvs"] += 1
            elif type(pv) is SummationPV:
                num_pvs_dict["num_summation_pvs"] += 1
            elif type(pv) is RefreshPV:
                num_pvs_dict["num_refresh_pvs"] += 1

        print("Virtac stats:")
        print(
            "\t Tune feedbacks is "
            f"{('disabled' if self._disable_tunefb else 'enabled')}"
        )
        print(
            "\t Emittance calculations are "
            f"{('disabled' if self._disable_emittance else 'enabled')}"
        )
        print(
            f"\t PV monitoring is {('enabled' if self._pv_monitoring else 'disabled')}"
        )
        print(f"\t Total pvs: {total_num_pvs}")
        print(f"\t\t CA pvs: {num_pvs_dict['num_ca_pvs']}")
        print(f"\t\t PV pvs: {num_pvs_dict['num_pvs']}")
        print(f"\t\t Readback pvs: {num_pvs_dict['num_readback_pvs']}")
        print(f"\t\t Direct pvs: {num_pvs_dict['num_direct_pvs']}")
        print(f"\t\t Offset pvs: {num_pvs_dict['num_offset_pvs']}")
        print(f"\t\t Monitor pvs: {num_pvs_dict['num_monitor_pvs']}")
        print(f"\t\t Collation pvs: {num_pvs_dict['num_collation_pvs']}")
        print(f"\t\t Inverse pvs: {num_pvs_dict['num_inverse_pvs']}")
        print(f"\t\t Summation pvs: {num_pvs_dict['num_summation_pvs']}")
        print(f"\t\t Refresh pvs: {num_pvs_dict['num_refresh_pvs']}")

        if verbosity >= 1:
            print("\tAvailable PVs")
            for pv in self._pv_dict.values():
                print(f"\t\t{pv.name}, {type(pv)}")
