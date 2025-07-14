import logging
import time
from dataclasses import dataclass

import cothread
import numpy
import pytac
from cothread.catools import caget, camonitor, caput
from softioc import builder


@dataclass
class RecordData:
    """Class for holding information required to create a softioc"""

    record_type: str
    lower: str | None = None
    upper: str | None = None
    precision: str | None = None
    drive_high: str | None = None
    drive_low: str | None = None
    zrvl: str | None = None
    zrst: str | None = None
    scan: str | None = "I/O Intr"
    always_update: str | None = False
    initial_value: int | float | numpy.typing.NDArray = 0


class PV:
    """Class that stores information about a PV and allows monitoring and writing to a
    PV over channel access"""

    def __init__(self, name: str, record_data: RecordData):
        logging.debug(f"Creating PV {name}")
        self.name: str = name
        self._tune_feedback_enabled: bool = False
        self._record = None
        self._pytac_elements: list = []
        self._pytac_field = None
        self.update_lattice = False
        if record_data is not None:
            self.create_softioc_record(record_data)

    def _on_update(self, value, name):
        """The callback function called when the PV updates.

        This functions needs to be kept FAST as it can be called rapidly by CA clients.

        Args:
            value (number): The value that has just been set to the record.
            name (str): The name of record object that has just been set to.
        """
        logging.info(f"PV {name} changed to: {value}")

    def set_tune_feedback_enabled(self, enable):
        logging.debug(f"Setting tunefb enabled to: {enable} for PV: {self.name}")
        self._tune_feedback_enabled = enable

    def get_pytac_data(self):
        return self._pytac_elements, self._pytac_field

    def append_pytac_element(self, element):
        self._pytac_elements.append(element)

    def set_pytac_field(self, field):
        self._pytac_field = field

    def set_record_field(self, field, value):
        self._record.set_field(field, value)

    def create_softioc_record(
        self,
        record_data: RecordData,
    ):
        logging.debug(f"Creating softioc record {self.name}")
        if record_data.record_type == "ai":
            self._record = builder.aIn(
                self.name,
                PREC=record_data.precision,
                LOPR=record_data.lower,
                HOPR=record_data.upper,
                MDEL="-1",
                SCAN=record_data.scan,
                initial_value=record_data.initial_value,
            )
        elif record_data.record_type == "ao":
            self._record = builder.aOut(
                self.name,
                PREC=record_data.precision,
                LOPR=record_data.lower,
                DRVH=record_data.drive_high,
                DRVL=record_data.drive_low,
                HOPR=record_data.upper,
                MDEL="-1",
                initial_value=record_data.initial_value,
                always_update=record_data.always_update,
                on_update_name=self._on_update,
            )
        elif record_data.record_type == "wfm":
            self._record = builder.WaveformOut(
                self.name,
                initial_value=record_data.initial_value,
                always_update=True,
                SCAN=record_data.scan,
            )
        elif record_data.record_type == "mbbi":
            self._record = builder.mbbIn(
                self.name,
                initial_value=record_data.initial_value,
                ZRVL=record_data.zrvl,
                ZRST=record_data.zrst,
                SCAN=record_data.scan,
            )
        else:
            raise ValueError(
                f"Failed to create PV with record type: {record_data.record_type}"
            )

    def get(self):
        return self._record.get()

    def set(self, value, index=None):
        """
        Args:
            value (number): The value to set to the PV.
        """
        logging.debug(f"PV: {self.name} changed to: {value}")
        return self._record.set(value)


class DirectPV(PV):
    """When this PV has its value updated, we do two things. We update its
    paired in_record with value"""

    def __init__(self, name, record_data: RecordData, in_record: PV):
        super().__init__(name, record_data)
        self._in_record = in_record
        self._offset_record = None

    def _on_update(self, value, name):
        """This function is called whenever this PVs softioc record processes. It
        in_record with the value that has been set to this PVs record (out_record) and
        then sets the value to the in_records Pytac elements.

        This functions needs to be kept FAST as it can be called rapidly by CA clients.

        Args:
            value (number): The value that has just been set to the record.
            name (str): The name of record object that has just been set to.
        """
        logging.debug("Read value %s on pv %s", value, name)
        self._in_record.set(value)
        if self._tune_feedback_enabled is True and self._offset_record is not None:
            try:
                offset = self._offset_record.get()
                value += offset
            except KeyError as e:
                print(e)

        elements, field = self._in_record.get_pytac_data()
        for element in elements:
            # Some elements such as bend magnets share a single PV which is used to
            # update them all to the same value
            logging.debug(
                f"Updating lattice for pv: {self._in_record.name} to val {value}"
            )
            element.set_value(
                field,
                value,
                units=pytac.ENG,
                data_source=pytac.SIM,
            )

    def attach_offset_record(self, offset_record: PV):
        self._offset_record = offset_record
        logging.debug(f"Attaching offset record: {offset_record} to PV: {self.name}")


class MonitorPV(PV):
    """This type of PV monitors one or more PVs using channal access and does a callback
    when one of the monitors returns"""

    def __init__(
        self, name, record_data: RecordData, monitored_records: list[PV], callbacks=None
    ):
        super().__init__(name, record_data)
        self._monitored_records = list(monitored_records)
        # Stores a list of tuples containing the name of the monitored pv and its
        # associated callback function
        self._monitor_list: list[tuple] = []
        self._camonitor_handles = []
        self._timeout: float
        if callbacks is None:
            callbacks = [self.set]
        self.monitor_pvs(self._monitored_records, callbacks)

    def monitor_pvs(self, pvs: list[PV], callbacks: list):
        """Get a list of PVs and a list of callbacks. If only a single callback is
        supplied, then it is used for each PV, if a list of callbacks is supplied, then
        it must be the same length as the list of pvs"""
        if len(callbacks) > 1:
            assert len(pvs) == len(callbacks)

        if len(callbacks) == 1:
            # Use the same callback for all pvs
            pv_names = []
            callback = callbacks[0]
            for pv in pvs:
                self._monitor_list.append((pv, callback))
                pv_names.append(pv.name)
            self._camonitor_handles.extend(camonitor(pv_names, callback))
        else:
            for pv, callback in zip(pvs, callbacks, strict=True):
                self._monitor_list.append((pv, callback))
                self._camonitor_handles.append(camonitor(pv.name, callback))

    def toggle_monitoring(self, enable):
        """Can be used to switch off this PVs monitoring by closing camonitor
        subscriptions or to re-enable monitoring by recreating the subscriptions."""
        if enable:
            logging.debug(f"Enabling monitoring for PV {self.name}")
            pv_list = []
            callback_list = []
            for pv, callback in self._monitor_list:
                pv_list.append(pv)
                callback_list.append(callback)
            self.monitor_pvs(pv_list, callback_list)
        else:
            logging.debug(f"Disabling monitoring for PV {self.name}")
            for handle in self._camonitor_handles:
                handle.close()


class RefreshPV(PV):
    """This record is currently very convoluted and needs refactoring. Currently we
    hijack an already created PV and steal its data and then replace it in the pv
    dictionary. The actual functionality of this pv is: we monitor the monitor_record
    and when it changes, we update this record with the value returned and then poke
    a third record, the refresh_record, which then processes, during which it will
    get the value stored in this record as part of its update."""

    def __init__(
        self,
        name,
        monitor_record: PV,
        refresh_record: PV,
        pytac_elements,
        pytac_field,
        record,
    ):
        super().__init__(name, None)
        # TODO: We steal an already created record from a previously made PV and then
        # replace that pv when we should only need to create this PV once.
        self._record = record
        self._pytac_field = pytac_field
        self._pytac_elements = pytac_elements
        self._refresh_record = refresh_record
        camonitor(monitor_record, self.set)

    def set(self, value):
        """An imitation  of the set method of Soft-IOC records, that applies a
        transformation to the value before setting it to the output record.
        """
        logging.debug(
            f"Setting refresh record: {self.name} to {value} and prodding "
            f"{self._refresh_record._record.name} to process "
        )
        # print(f"{self.name} Refreshing quad")
        # print(f"Setting PV {self.name} to {value}")
        self._record.set(value)
        # print(f"{self._refresh_record._record} Proccing SETI PV")
        self._refresh_record._record.set_field("PROC", 1)


class InversePV(MonitorPV):
    """Used to invert a boolean array waveform 'in_record', ie swap true to false and
    false to true and then save the result in its own waveform _record."""

    def __init__(self, name, record_data: RecordData, in_record: list[PV]):
        super().__init__(name, record_data, in_record, [self.set])
        self.name = name

    def set(self, value, index=None):
        """An imitation  of the set method of Soft-IOC records, that applies a
        transformation to the value before setting it to the output record.
        """
        logging.debug(f"{self.name} Inverting data")
        value = numpy.asarray(value, dtype=bool)
        value = numpy.asarray(numpy.invert(value), dtype=int)
        self._record.set(value)


class SummationPV(MonitorPV):
    """Sum a list of PV values and set this PVs record to the result"""

    def __init__(self, name, record_data: RecordData, in_records: list[PV]):
        super().__init__(name, record_data, in_records, [self.summate])

    def summate(self, value, index=None):
        """An imitation  of the set method of Soft-IOC records."""
        value = sum([pv.get() for pv in self._monitored_records])
        # print(self._in_records)
        # print(f"{self.name} Summing data, value: {value}")
        self._record.set(value)


class CollationPV(MonitorPV):
    """Monitor a list of input PVs and when any change then get the values of all PVs
    and collate them into a numpy array which is then set to the record owned by this
    PV."""

    def __init__(self, name, record_data: RecordData, in_records: list[PV]):
        super().__init__(name, record_data, in_records, [self.set_update_required])
        self._last_update_time = time.time()
        self._update_required = False
        cothread.Spawn(self.periodic_update)

    def periodic_update(self):
        while True:
            if self._update_required:
                self.collate()
            else:
                cothread.Sleep(0.5)

    def set_update_required(self, value, index=None):
        # print("Camonitor returning")
        self._update_required = True

    def collate(self):
        """An imitation of the set method of Soft-IOC records."""
        logging.debug("Collating data")
        if time.time() - self._last_update_time < 0.5:
            cothread.Sleep(time.time() - self._last_update_time)
        value = numpy.array([record.get() for record in self._monitored_records])
        self._last_update_time = time.time()
        self._record.set(value)
        self._record.set_field("PROC", 1)
        self._update_required = False


class CaPV:
    """Uses channel access to get and set an EPICS PV to a value. This PV does not need
    a softioc record."""

    def __init__(self, name: str):
        self.name = name

    def get(self):
        logging.debug(f"{self.name} caget, value: {caget(self._caput_pv_name)}")
        return caget(self.name)

    def set(self, value, index=None):
        """
        Args:
            value (number): The value to caput to the PV.
        """
        print(f"{self.name} Caputting data")
        return caput(self.name, value)
