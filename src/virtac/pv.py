import logging

import numpy
import pytac
from cothread.catools import caget, caput
from softioc import builder


class PV:
    """Class that stores information about a PV and allows monitoring and writing to a
    PV over channel access"""

    def __init__(self, name: str, tune_feedback: bool = False):
        self._pv_name: str = name
        self._tune_feedback_status: bool = tune_feedback
        self._record = None
        self._pytac_elements: list = []
        self._pytac_field = None
        self._timeout: float

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

    def append_pytac_element(self, element):
        self._pytac_elements.append(element)

    def set_pytac_field(self, field):
        self._pytac_field = field

    def create_softioc_record(
        self,
        record_type,
        lower,
        upper,
        precision,
        drive_high,
        drive_low,
        initial_value,
        zrvl=None,
        zrst=None,
        pini=None,
        always_update=True,
    ):
        if record_type == "ai" or record_type == "aIn":
            self._record = builder.aIn(
                self._pv_name,
                PREC=precision,
                LOPR=lower,
                HOPR=upper,
                MDEL="-1",
                initial_value=initial_value,
            )
        elif record_type == "ao" or record_type == "aOut":
            self._record = builder.aOut(
                self._pv_name,
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
        elif record_type == "wfm" or record_type == "Waveform":
            self._record = builder.WaveformOut(
                self._pv_name,
                initial_value=initial_value,
                always_update=True,
            )
        elif record_type == "mbbi":
            self._record = builder.mbbIn(
                self.name,
                initial_value=initial_value,
                ZRVL=zrvl,
                ZRST=zrst,
                PINI=pini,
            )
        else:
            raise ValueError(f"Failed to create PV with record type: {record_type}")


class DumbPV(PV):
    """Stores a passive softioc record which can be updated, but does nothing
    when it is updated, other than update its own value."""

    def __init__(self, name: str):
        super().__init__(name)
        self._record = None


class DirectPV(PV):
    """When this PV has its value updated, we find its paired PV and update it based
    on simulation data."""

    def __init__(self, name, in_record: PV):
        super().__init__(name)
        self._record = None
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
        self._in_record._record.set(value)

        if self._tune_feedback_status is True:
            try:
                value += self.offset_record._record.get()
            except KeyError:
                pass

        for element in self._in_record._pytac_elements:
            element.set_value(
                self._in_record._pytac_field,
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
        """
        Args:
            value (number): The value to caput to the PV.
        """
        return caget(self._caput_pv_name)

    def set(self, value):
        """
        Args:
            value (number): The value to caput to the PV.
        """
        return caput(self._caput_pv_name, value)


class InversePV(PV):
    """Used to invert a boolean array waveform 'in_record', ie swap true to false and
    false to true and then save the result in its own waveform _record."""

    def __init__(self, name, in_record: PV):
        super().__init__(name)
        self.name = name
        self._in_record = in_record

    def set(self, value):
        """An imitation  of the set method of Soft-IOC records, that applies a
        transformation to the value before setting it to the output record.
        """
        value = numpy.asarray(value, dtype=bool)
        value = numpy.asarray(numpy.invert(value), dtype=int)
        self._record.set(value)


class SummationPV(PV):
    """"""

    def __init__(self, name, in_records: list[PV]):
        super().__init__(name)
        self.name = name
        self._in_records = in_records

    def set(self, value):
        """An imitation  of the set method of Soft-IOC records."""
        value = sum([record.get() for record in self._in_records])
        self._record.set(value)


class CollationPV(PV):
    """"""

    def __init__(self, name, in_records: list[PV]):
        super().__init__(name)
        self.name = name
        self._in_records = in_records

    def set(self, value):
        """An imitation  of the set method of Soft-IOC records."""
        value = numpy.array([record.get() for record in self._in_records])
        self._record.set(value)
