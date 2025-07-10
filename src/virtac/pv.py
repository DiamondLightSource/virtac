import logging
import time

import cothread
import numpy
import pytac
from cothread.catools import caget, camonitor, caput
from softioc import builder


class PV:
    """Class that stores information about a PV and allows monitoring and writing to a
    PV over channel access"""

    def __init__(self, name: str, tune_feedback: bool = False):
        self.name: str = name
        print(f"Creating record {name}")
        self._tune_feedback_status: bool = tune_feedback
        self._record = None
        self._pytac_elements: list = []
        self._pytac_field = None
        self._monitor_list: list = []
        self._camonitor_handle = None
        self._camonitor_callback = self._monitor_callback
        self._timeout: float
        self.update_lattice = False

    def _on_update(self, value, name):
        """The callback function called when the PV updates.

        This functions needs to be kept FAST as it can be called rapidly by CA clients.

        Args:
            value (number): The value that has just been set to the record.
            name (str): The name of record object that has just been set to.
        """
        logging.info(f"PV {name} changed to: {value}")

    def set_tune_feedback_status(self, status):
        self._tune_feedback_status = status

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
        record_type,
        lower=None,
        upper=None,
        precision=None,
        drive_high=None,
        drive_low=None,
        zrvl=None,
        zrst=None,
        scan="I/O Intr",
        always_update=True,
        initial_value=0,
    ):
        if record_type == "ai":
            self._record = builder.aIn(
                self.name,
                PREC=precision,
                LOPR=lower,
                HOPR=upper,
                MDEL="-1",
                SCAN=scan,
                initial_value=initial_value,
            )
        elif record_type == "ao":
            self._record = builder.aOut(
                self.name,
                PREC=precision,
                LOPR=lower,
                DRVH=drive_high,
                DRVL=drive_low,
                HOPR=upper,
                MDEL="-1",
                initial_value=initial_value,
                on_update_name=self._on_update,
                always_update=always_update,
            )
        elif record_type == "wfm":
            self._record = builder.WaveformOut(
                self.name,
                initial_value=initial_value,
                always_update=True,
                SCAN=scan,
            )
        elif record_type == "mbbi":
            self._record = builder.mbbIn(
                self.name,
                initial_value=initial_value,
                ZRVL=zrvl,
                ZRST=zrst,
                SCAN=scan,
            )
        else:
            raise ValueError(f"Failed to create PV with record type: {record_type}")

    def monitor_pvs(self, pv_names):
        self._monitor_list.extend(pv_names)
        self._camonitor_handle = camonitor(self._monitor_list, self._camonitor_callback)

    def _monitor_callback(self, value, index):
        print(f"Monitored PV: {self._monitor_list[index]} updated. New value: {value}")

    def get(self):
        return self._record.get()

    def set(self, value):
        """
        Args:
            value (number): The value to set to the PV.
        """
        logging.debug(f"PV setter: {self.name} changed to: {value}")
        return self._record.set(value)


class DumbPV(PV):
    """Stores a passive softioc record which can be updated, but does nothing
    when it is updated, other than update its own value."""

    def __init__(self, name: str):
        super().__init__(name)


class DirectPV(PV):
    """When this PV has its value updated, we find its paired PV and update it based
    on simulation data."""

    def __init__(self, name, in_record: PV):
        super().__init__(name)
        self._in_record = in_record
        self._offset_record = None

    def _on_update(self, value, name):
        """The callback function passed to out records, it is called after
        successful record processing has been completed. It updates the out
        record's corresponding in record with the value that has been set and
        then sets the value to the Pytac lattice.

        This functions needs to be kept FAST as it can be called rapidly by CA clients.

        Args:
            value (number): The value that has just been set to the record.
            name (str): The name of record object that has just been set to.
        """
        logging.debug("Read value %s on pv %s", value, name)
        self._in_record.set(value)

        if self._tune_feedback_status is True:
            try:
                value += self.offset_record.get()
            except KeyError:
                pass

        elements, field = self._in_record.get_pytac_data()
        for element in elements:
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


class CaputPV(PV):
    """Does a caput to set a remote EPICS PV to a value. This PV does not need
    a softioc record."""

    def __init__(self, name: str, caput_pv_name: str):
        super().__init__(name)
        self._caput_pv_name = caput_pv_name

    def get(self):
        logging.debug(f"{self.name} caget, value: {caget(self._caput_pv_name)}")
        return caget(self._caput_pv_name)

    def set(self, value):
        """
        Args:
            value (number): The value to caput to the PV.
        """
        print(f"{self.name} Caputting data")
        return caput(self._caput_pv_name, value)


class InversePV(PV):
    """Used to invert a boolean array waveform 'in_record', ie swap true to false and
    false to true and then save the result in its own waveform _record."""

    def __init__(self, name, in_record: PV):
        super().__init__(name)
        self.name = name
        self._in_record = in_record
        self._camonitor_callback = self.set
        self.monitor_pvs(self._in_record[0].name)

    def set(self, value):
        """An imitation  of the set method of Soft-IOC records, that applies a
        transformation to the value before setting it to the output record.
        """
        logging.debug(f"{self.name} Inverting data")
        value = numpy.asarray(value, dtype=bool)
        value = numpy.asarray(numpy.invert(value), dtype=int)
        self._record.set(value)


class SummationPV(PV):
    """"""

    def __init__(self, name, in_records: list[PV]):
        super().__init__(name)
        self.name = name
        self._in_records = in_records
        self._camonitor_callback = self.set
        self.monitor_pvs(pv.name for pv in self._in_records)

    def set(self, value, index):
        """An imitation  of the set method of Soft-IOC records."""
        value = sum([pv.get() for pv in self._in_records])
        # print(self._in_records)
        # print(f"{self.name} Summing data, value: {value}")
        self._record.set(value)


class CollationPV(PV):
    """Monitor a list of input PVs and when any change then get the values of all PVs
    and collate them into a numpy array which is then set to the record owned by this
    PV."""

    def __init__(self, name, in_records: list[PV]):
        super().__init__(name)
        self.name = name
        self._in_records = in_records
        self._update_required = False
        self._camonitor_callback = self.set_update_required
        self._last_update_time = time.time()
        self.monitor_pvs([pv.name for pv in self._in_records])
        cothread.Spawn(self.periodic_update)

    def periodic_update(self):
        while True:
            # print("Running periodic task")
            if self._update_required:
                self.set()
            else:
                cothread.Sleep(0.5)

    def set_update_required(self, value, index):
        # print("Camonitor returning")
        self._update_required = True

    def set(self):
        """An imitation of the set method of Soft-IOC records."""
        # Temporary limit to stop this updating when every in_record updates
        # print(f"{self.name} Collating data")
        if time.time() - self._last_update_time < 0.5:
            cothread.Sleep(time.time() - self._last_update_time)
        value = numpy.array([record.get() for record in self._in_records])
        logging.debug("Collate data")
        self._last_update_time = time.time()
        self._record.set(value)
        self._record.set_field("PROC", 1)
        self._update_required = False
