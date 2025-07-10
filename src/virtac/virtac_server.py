import csv
import logging
import typing
from warnings import warn

import atip
import numpy
import pytac
from cothread.catools import camonitor
from pytac.device import SimpleDevice
from pytac.exceptions import FieldException, HandleException

from .pv import (
    PV,
    CaPV,
    CollationPV,
    DirectPV,
    InversePV,
    MonitorPV,
    RefreshPV,
    SummationPV,
)


class VirtacServer:
    """The soft-ioc server which contains the configuration and PVs for the VIRTAC.
    It allows ATIP to be interfaced using EPICS, in the same manner as the live machine.

    **Attributes**

    Attributes:
        lattice (pytac.lattice.Lattice): An instance of a Pytac lattice with a
                                          simulator data source.
        tune_feedback_status (bool): A boolean indicating whether the tune
                                      feedback records have been created and
                                      the monitoring systems are running.
    .. Private Attributes:
           _pv_monitoring (bool): Whether the mirrored PVs are being monitored.
           _tune_fb_csv_path (str): The path to the tune feedback .csv file.
           _in_records (dict): A dictionary containing all the created in
                                records, a list of associated element indexes and
                                Pytac field, i.e. {in_record: [[index], field]}.
           _out_records (dict): A dictionary containing the names of all the
                                 created out records and their associated in
                                 records, i.e. {out_record.name: in_record}.
           _rb_only_records (list): A list of all the in records that do not
                                     have an associated out record.
           _feedback_records (dict): A dictionary containing all the feedback
                                      related records, in the same format as
                                      _in_records because they are all readback
                                      only.
           _mirrored_records (dict): A dictionary containing the PVs that the
                                      mirrored records monitor for a change
                                      and the associated mirror, in the form
                                      {monitored PV: mirror record/object}.
           _monitored_pvs (dict): A dictionary of all the PVs that are being
                                   monitored for a change and the associated
                                   camonitor object, in the form
                                   {monitored PV: camonitor object}.
           _offset_pvs (dict): A dictionary of the PVs to apply offset to and
                                their associated offset records from which to
                                get the offset from.
            _record_names (dict[str: softioc.builder.record]): A dictonary
                                containing the name of every pv created by the
                                virtual accelerator and the pv object itself.
    """

    def __init__(
        self,
        ring_mode,
        limits_csv=None,
        bba_csv=None,
        feedback_csv=None,
        mirror_csv=None,
        tune_csv=None,
        disable_emittance=False,
    ):
        """
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
            disable_emittance (bool): Whether the emittance should be disabled.
        """
        self.lattice = atip.utils.loader(ring_mode, self.update_pvs, disable_emittance)
        self.tune_feedback_status = False
        self._pv_monitoring = False
        self._tune_fb_csv_path = tune_csv
        self._pv_dict: dict[str, PV] = {}
        print("Starting record creation.")
        self._create_records(limits_csv, disable_emittance)
        if bba_csv is not None:
            self._create_bba_records(bba_csv)
        if feedback_csv is not None:
            self._create_feedback_records(feedback_csv, disable_emittance)
        if mirror_csv is not None:
            self._create_mirror_records(mirror_csv)
        print(f"Finished creating all {len(self._pv_dict)} records.")

    def update_pvs(self):
        """The callback function passed to ATSimulator during lattice creation,
        it is called each time a calculation of physics data is completed. It
        updates all the in records that do not have a corresponding out record
        with the latest values from the simulator.
        """
        logging.debug("Updating output PVs")
        logging.debug("Building list")
        things = []
        for name, pv in self._pv_dict.items():
            if pv.update_lattice:
                things.append((name, pv))
        logging.debug("Doing updates")
        for name, pv in things:
            logging.debug(f"Updating pv {name}")
            if pv.update_lattice:
                elements, field = pv.get_pytac_data()
                # elements is a list of pytac lattice elements associated with a pv. The
                # lattice itself is also considered an element, and can be associated with
                # pvs.
                # for element in elements:
                #     # TODO: Does this make sense? A PV can have multiple elements, but what
                #     # is the point of looping through them and updating the PVs value to
                #     # each elements value? Only the last element in the list will have an
                #     # effect?
                try:
                    value = elements[0].get_value(
                        field, units=pytac.ENG, data_source=pytac.SIM
                    )
                    logging.debug(f"Update_pvs: {name} to val {value}")
                    pv.set(value)
                except FieldException:
                    print("Missing pytac field")
                    # print(e)
        logging.debug("Finished updating output PVs")

    def _create_records(self, limits_csv, disable_emittance):
        """Create all the standard records from both lattice and element Pytac
        fields. Several assumptions have been made for simplicity and
        efficiency, these are:
            - That bend elements all share a single PV, and are the only
               element family to do so.
            - That every field that has an out type record (SP) will also have
               an in type record (RB).
            - That all lattice fields are never setpoint and so only in records
               need to be created for them.

        Args:
            limits_csv (str): The filepath to the .csv file from which to
                                    load the pv limits.
            disable_emittance (bool): Whether the emittance related PVs should be
                                        created or not.
        """
        limits_dict = {}
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
                        str(line["refresh"]),
                    )

        bend_in_record = None
        for element in self.lattice:
            # There is only 1 bend PV in the lattice, if it has already been defined and
            # we have another bend element, then just register this element with the
            # existing pv. Otherwise create a new PV for the element
            if element.type_.upper() == "BEND" and bend_in_record is not None:
                bend_in_record.append_pytac_element(element)
            else:
                for field in element.get_fields()[pytac.SIM]:
                    value = element.get_value(
                        field, units=pytac.ENG, data_source=pytac.SIM
                    )
                    get_pv_name = element.get_pv_name(field, pytac.RB)
                    upper, lower, precision, drive_high, drive_low, refresh = (
                        limits_dict.get(
                            get_pv_name, (None, None, None, None, None, None)
                        )
                    )
                    in_pv = PV(get_pv_name)
                    in_pv.create_softioc_record(
                        "ai",
                        lower=lower,
                        upper=upper,
                        precision=precision,
                        drive_high=drive_high,
                        drive_low=drive_low,
                        initial_value=value,
                        scan=refresh,
                    )
                    in_pv.append_pytac_element(element)
                    in_pv.set_pytac_field(field)
                    in_pv.update_lattice = True
                    self._pv_dict[get_pv_name] = in_pv

                    try:
                        set_pv_name = element.get_pv_name(field, pytac.SP)
                    except HandleException:
                        pass
                    else:
                        upper, lower, precision, drive_high, drive_low, refresh = (
                            limits_dict.get(
                                get_pv_name, (None, None, None, None, None, None)
                            )
                        )
                        out_pv = DirectPV(set_pv_name, in_pv)
                        out_pv.create_softioc_record(
                            "ao",
                            lower=lower,
                            upper=upper,
                            precision=precision,
                            drive_high=drive_high,
                            drive_low=drive_low,
                            initial_value=value,
                        )
                        # out_pv.append_pytac_element(element)
                        # out_pv.set_pytac_field(field)

                        self._pv_dict[set_pv_name] = out_pv
                        if element.type_.upper() == "BEND" and bend_in_record is None:
                            bend_in_record = in_pv

        # Now for lattice fields.
        lat_fields = self.lattice.get_fields()
        lat_fields = set(lat_fields[pytac.LIVE]) & set(lat_fields[pytac.SIM])
        if disable_emittance:
            lat_fields -= {"emittance_x", "emittance_y"}
        for field in lat_fields:
            # Ignore basic devices as they do not have PVs.
            if not isinstance(self.lattice.get_device(field), SimpleDevice):
                get_pv_name = self.lattice.get_pv_name(field, pytac.RB)
                upper, lower, precision, drive_high, drive_low, refresh = (
                    limits_dict.get(get_pv_name, (None, None, None, None, None, None))
                )
                value = self.lattice.get_value(
                    field, units=pytac.ENG, data_source=pytac.SIM
                )
                in_pv = PV(get_pv_name)
                in_pv.create_softioc_record(
                    "ai",
                    lower=lower,
                    upper=upper,
                    precision=precision,
                    scan=refresh,
                    initial_value=value,
                )
                in_pv.append_pytac_element(self.lattice)
                in_pv.set_pytac_field(field)
                in_pv.update_lattice = True
                self._pv_dict[get_pv_name] = in_pv
        print("~*~*Woah, we're halfway there, Wo-oah...*~*~")

    def _create_bba_records(self, bba_csv):
        """Create all the beam-based-alignment records from the .csv file at the
        location passed, see create_csv.py for more information.

        Args:
            bba_csv (str): The filepath to the .csv file to load the
                                    records in accordance with.
        """
        self._create_feedback_or_bba_records_from_csv(bba_csv)

    def _create_feedback_records(self, feedback_csv, disable_emittance):
        """Create all the feedback records from the .csv file at the location
        passed, see create_csv.py for more information; records for one edge
        case are also created.

        Args:
            feedback_csv (str): The filepath to the .csv file to load the
                                    records in accordance with.
            disable_emittance (bool): Whether the emittance related PVs should be
                                        created or not.
        """
        # Create standard records from csv
        self._create_feedback_or_bba_records_from_csv(feedback_csv)

        # We can choose to not calculate emittance as it is not always required,
        # which decreases computation time.
        if not disable_emittance:
            # Special case: EMIT STATUS for the vertical emittance feedback
            name = "SR-DI-EMIT-01:STATUS"
            emit_status_pv = PV(name)
            emit_status_pv.create_softioc_record(
                "mbbi",
                zrvl=0,
                zrst="Successful",
            )
            self._pv_dict[name] = emit_status_pv

    def _create_feedback_or_bba_records_from_csv(self, csv_file):
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
                    # Here we convert that into a numpy array for initialising the
                    # record
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
                    pv = PV(name)
                    # TODO: Add some checks of csv data here?
                    pv.create_softioc_record(line["record_type"], initial_value=val)
                    pv.set_pytac_field(line["field"])
                    pv.append_pytac_element(self.lattice[int(line["index"]) - 1])
                    self._pv_dict[name] = pv

    def _create_mirror_records(self, mirror_csv):
        # Create a dictionary of monitored_pv: [pvs_affected]
        """Create all the mirror records from the .csv file at the location
        passed, see create_csv.py for more information.

        Args:
            mirror_csv (str): The filepath to the .csv file to load the
                                    records in accordance with.
        """
        with open(mirror_csv) as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                # Parse arguments.
                # Get a list of input pvs, these are all virtac owned pvs
                input_pvs = line["in_pv"].split(", ")
                if (len(input_pvs) > 1) and (
                    line["mirror_type"] in ["basic", "inverse"]
                ):
                    raise IndexError(
                        "Transformation and basic mirror types take only one input PV."
                    )
                elif (len(input_pvs) < 2) and (
                    line["mirror_type"] in ["collate", "summate"]
                ):
                    raise IndexError(
                        "collation and summation mirror types take at least two input "
                        "PVs."
                    )
                # Convert input pvs to record objects
                input_records = []
                for pv in input_pvs:
                    try:
                        # Lookup pv in our dictionary of softioc records
                        input_records.append(self._pv_dict[pv])
                    except KeyError:
                        # If not owned by us, then we get it from CA
                        input_records.append(CaPV(pv, pv))
                # Update the mirror dictionary.
                try:
                    # Waveform records may have values stored as a list such as: [5 1 3]
                    # Here we convert that into a numpy array for initialising the
                    # record
                    # TODO: This can be improved, the value data format should be the
                    # same as in other csv files
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
                    if line["mirror_type"] == "basic":
                        output_pv = MonitorPV(out_pv_name, input_records)
                        output_pv.create_softioc_record(
                            line["output_type"], initial_value=val, scan=line["refresh"]
                        )
                    elif line["mirror_type"] == "inverse":
                        output_pv = InversePV(out_pv_name, input_records)
                        output_pv.create_softioc_record(
                            line["output_type"], initial_value=val, scan=line["refresh"]
                        )
                    elif line["mirror_type"] == "summate":
                        output_pv = SummationPV(out_pv_name, input_records)
                        output_pv.create_softioc_record(
                            line["output_type"], initial_value=val, scan=line["refresh"]
                        )
                    elif line["mirror_type"] == "collate":
                        # print(out_pv_name, input_records, val)
                        output_pv = CollationPV(out_pv_name, input_records)
                        output_pv.create_softioc_record(
                            line["output_type"], initial_value=val, scan=line["refresh"]
                        )
                    else:
                        raise TypeError(
                            f"{line['mirror type']} is not a valid mirror type; please "
                            "enter a currently supported type from: 'basic', 'summate',"
                            " 'collate' and 'inverse'."
                        )

                    self._pv_dict[out_pv_name] = output_pv

    def setup_tune_feedback(self, tune_csv=None):
        """Read the tune feedback .csv and find the associated offset PVs,
        before starting monitoring them for a change to mimic the behaviour of
        the quadrupoles used by the tune feedback system on the live machine.

        .. Note:: This is intended to be on the recieving end of the tune
           feedback system and doesn't actually perfom tune feedback itself.

        Args:
            tune_csv (str): A path to a tune feedback .csv file to be used
                             instead of the default filepath passed at startup.
        """
        if tune_csv is not None:
            self._tune_fb_csv_path = tune_csv
        if self._tune_fb_csv_path is None:
            raise ValueError(
                "No tune feedback .csv file was given at "
                "start-up, please provide one now; i.e. "
                "server.start_tune_feedback('<path_to_csv>')"
            )
        with open(self._tune_fb_csv_path) as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                set_record = self._pv_dict[line["set_pv"]]
                old_offset_record = self._pv_dict[line["offset_pv"]]
                elements, field = old_offset_record.get_pytac_data()
                new_offset_record = RefreshPV(
                    line["offset_pv"],
                    line["delta_pv"],
                    set_record,
                    elements,
                    field,
                    old_offset_record._record,
                )
                self._pv_dict[line["offset_pv"]] = new_offset_record
                set_record.attach_offset_record(new_offset_record)
                set_record.set_tune_feedback_status(True)

    # Needs fixing from refactor
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
