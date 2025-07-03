import logging
from collections.abc import Callable
from dataclasses import dataclass

import pytac
from cothread.catools import caget, camonitor, caput
from softioc import builder


class PV:
    """Class that stores information about a PV and allows monitoring and writing to a PV \
        over channel access"""

    def __init__(self, name: str):
        self._pv_name: str = name
        self._tune_feedback_status: bool
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


class DumbPV(PV):
    """Stores a passive softioc record which can be updated, but does nothing
    when it is updated, other than update its own value."""

    def __init__(self, name: str):
        super().__init__(name)
        self._record = None

    def create_softioc_record(
        self, type, lower, upper, precision, drive_high, drive_low, initial_value
    ):
        if type == "ai":
            self._record = builder.aIn(
                self._pv_name,
                PREC=precision,
                LOPR=lower,
                HOPR=upper,
                MDEL="-1",
                initial_value=initial_value,
            )


class DirectPV(PV):
    """When this PV has its value updated, we find its paired PV and update it based
    on simulation data."""

    def __init__(self, name, in_record):
        super().__init__(name)
        self._record = None
        self._in_record = in_record

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
                offset_record = self._offset_pvs[name]
                value += offset_record.get()
            except KeyError:
                pass

        for i in self.index:
            self.lattice[i - 1].set_value(
                self.field, value, units=pytac.ENG, data_source=pytac.SIM
            )

    def create_softioc_record(
        self, type, lower, upper, precision, drive_high, drive_low, initial_value
    ):
        if type == "ao":
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
                always_update=True,
            )
