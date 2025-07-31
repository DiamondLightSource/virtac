import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import cothread
import numpy
import pytac
from cothread.catools import _Subscription, ca_nothing, caget, camonitor, caput
from softioc import builder
from softioc.pythonSoftIoc import RecordWrapper

RecordValue: TypeAlias = int | float | numpy.typing.NDArray
PytacItem: TypeAlias = pytac.lattice.Lattice | pytac.element.Element


@dataclass
class RecordData:
    """Class for holding information required to create a softioc record"""

    record_type: str
    lower: str | None = None
    upper: str | None = None
    precision: str | None = None
    drive_high: str | None = None
    drive_low: str | None = None
    zrvl: str | None = None
    zrst: str | None = None
    scan: str = "I/O Intr"
    pini: str = "YES"
    always_update: bool | None = False
    initial_value: RecordValue = 0

    def __post_init__(self):
        if not isinstance(self.record_type, str):
            raise ValueError("Record field `record_type` must be of integer type")
        if not isinstance(self.scan, str):
            raise ValueError("Record field `scan` must be of integer type")
        if not isinstance(self.pini, str):
            raise ValueError("Record field `pini` must be of integer type")


class PV:
    """Stores variables and functions related to an EPICS PV which
    the VIRTAC requires to control the PV.

    Args:
        name (str): Used to set self.name
        record_data (RecordData | None): Dataclass used to create this PVs softioc
            record.

    Attributes:
        self.name (str): The name used to get both the PV and its softioc record.
        self._record (softioc.pythonSoftIoc.RecordWrapper): This softioc record is the
            heart of the PV class, the main purpose of PV objects is to manage the
            setting and getting of these records.
        self._pytac_items (list[PytacItem]): An optional list of pytac elements or the
            pytac lattice itself which can be set/get from this PV.
        self._pytac_field (str): The field on the element(s) to set/get

    """

    def __init__(self, name: str, record_data: RecordData | None):
        logging.debug(f"Creating PV {name}")
        self.name: str = name
        self._record: RecordWrapper = None
        self._pytac_items: list[PytacItem] = []
        self._pytac_field: str = ""
        if record_data is not None:
            self.create_softioc_record(record_data)

    def _on_update(self, value: RecordValue, name: str):
        """The callback function called when the softioc record updates.

        This functions needs to be kept FAST as it can be called rapidly by CA clients.

        Args:
            value (RecordValue): The value that has just been set to the record.
            name (str): The name of the softioc record that has just been set to.
        """
        logging.info(f"PV {name} changed to: {value}")

    def get_pytac_data(self) -> tuple[list[PytacItem], str]:
        """Return the list of pytac elements and the field defined for this PV"""
        return self._pytac_items, self._pytac_field

    def append_pytac_item(self, pytac_item: PytacItem):
        """Append a pytac item to the list of pytac items defined for this PV

        Args:
            pytac_item (list[PytacItem]): The pytac element or lattice to append."""
        self._pytac_items.append(pytac_item)

    def set_pytac_field(self, field: str):
        """Set this PVs pytac field to the passed value

        Args:
            field (str): The pytac field to the value to."""
        self._pytac_field = field

    def set_record_field(self, field: str, value: str | RecordData):
        """Set a field on this PVs softioc record

        Args:
            field (softioc.field): The EPICS field to set on the softioc record
            value (str | RecordData): The value to set to the EPICS field"""
        self._record.set_field(field, value)

    def create_softioc_record(
        self,
        record_data: RecordData,
    ):
        """Create this PVs softioc record (self._record).

        Args:
            record_data (RecordData): Dataclass used to create this PVs softioc record.
        """

        logging.debug(f"Creating softioc record {self.name}")
        if record_data.record_type == "ai":
            self._record = builder.aIn(
                self.name,
                PREC=record_data.precision,
                LOPR=record_data.lower,
                HOPR=record_data.upper,
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
                initial_value=record_data.initial_value,
                always_update=record_data.always_update,
                on_update_name=self._on_update,
            )
        elif record_data.record_type == "wfm":
            self._record = builder.WaveformOut(
                self.name,
                initial_value=record_data.initial_value,
                PINI=record_data.pini,
                SCAN=record_data.scan,
                always_update=record_data.always_update,
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
                "Failed to create softioc record with record type: "
                f"{record_data.record_type}"
            )

    def get_record(self) -> RecordWrapper:
        """Return this PVs softioc record.

        Care should be taken when manipulating the returned record."""
        return self._record

    def get(self) -> RecordValue:
        """Get the value stored in this PVs softioc record"""
        return self._record.get()

    def set(self, value: RecordValue):
        """Set a value to this PVs softioc record.

        Args:
            value (RecordValue): The value to set to the softioc record.
        """
        logging.debug(f"PV: {self.name} changed to: {value}")
        self._record.set(value)


class ReadbackPV(PV):
    """This PV type designates that this PVs softioc record should be updated with the
    value from its connected pytac element(s) and field after a lattice recalculation.

    This PV class in itself does not automatically update, rather ReadbackPVs should be
    added to a list of PVs which are to be updated by using a callback. Such as in
    VirtacServer.update_pvs().
    TODO: Could we automatically have this functionality in this PV rather than relaying
    on an external callback?

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
    """

    def __init__(self, name: str, record_data: RecordData):
        super().__init__(name, record_data)


class SetpointPV(PV):
    """This PV is used to set a value to another PV and its associated pytac item(s).

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        in_pv (PV): The PV object to pass to _in_pv

    Attributes:
        _in_pv (PV): The PV which is to be updated when the SetpointPV's
            softioc record processes.
    """

    def __init__(self, name: str, record_data: RecordData, in_pv: PV):
        super().__init__(name, record_data)
        self._in_pv: PV = in_pv

    def _on_update(self, value: RecordValue, name: str):
        """This function sets value to self._in_pv._record and also sets value to the
        pytac item and field configured for self._in_pv.

        This function is called whenever this PVs softioc record processes. It needs to
        be kept FAST as it can be called rapidly by CA clients which are writing to
        self._record.

        Args:
            value (RecordValue): The value that has just been set to self._record.
            name (str): The name of self._record object.
        """
        logging.debug("Read value %s on pv %s", value, name)
        self._in_pv.set(value)

        # TODO: This functionality should really be done from the _in_records set
        # function.
        pytac_items, field = self._in_pv.get_pytac_data()
        for item in pytac_items:
            # Some elements such as bend magnets share a single PV which is used to
            # update them all to the same value
            logging.debug(f"Updating lattice for pv: {self._in_pv.name} to val {value}")
            item.set_value(
                field,
                value,
                units=pytac.ENG,
                data_source=pytac.SIM,
            )


class OffsetPV(SetpointPV):
    """This PV is similar to SetpointPV, except when it updates another PVs pytac item,
    it first gets an offset value from a third PV which is added to the value before
    setting.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        in_pv (PV): The PV object to pass to _in_pv
        offset_pv (PV | None): The PV object to pass to _offset_pv. It is optional to
            pass this at initialisation, but if not passed, it must be later attached
            using the attach_offset_record method.

    Attributes:
        _in_pv (PV): The PV which is to be updated when the SetpointPV's
            softioc record processes.
        _offset_pv (PV | None): The PV which we get a value from to use as an offset
            during _on_update.
    """

    def __init__(
        self,
        name: str,
        record_data: RecordData,
        in_pv: PV,
        offset_pv: PV | None = None,
    ):
        super().__init__(name, record_data, in_pv)
        self._offset_record: PV | None = offset_pv

    def _on_update(self, value: RecordValue, name: str):
        """This function sets value to self._in_pv._record and also sets value (with an
        additional offset from self._offset_pv) to the pytac item and field configured
        for self._in_pv.

        This function is called whenever this PVs softioc record processes. It needs to
        be kept FAST as it can be called rapidly by CA clients which are writing to
        self._record.

        Args:
            value (RecordValue): The value that has just been set to self._record.
            name (str): The name of self._record object.
        """
        logging.debug("Read value %s on pv %s", value, name)
        self._in_pv.set(value)

        offset: RecordValue = self._offset_record.get()
        value += offset

        elements, field = self._in_pv.get_pytac_data()
        for element in elements:
            # Some elements such as bend magnets share a single PV which is used to
            # update them all to the same value
            logging.debug(f"Updating lattice for pv: {self._in_pv.name} to val {value}")
            element.set_value(
                field,
                value,
                units=pytac.ENG,
                data_source=pytac.SIM,
            )

    def attach_offset_record(self, offset_pv: PV):
        """Used to configure this PV with an offset PV in situations where the offset
        was created after this PV.

        Args:
            offset_pv (PV): The PV object to be used during this PVs' records' on_update
            function."""
        logging.debug(f"Attaching offset record: {offset_pv} to PV: {self.name}")
        self._offset_record = offset_pv


class MonitorPV(PV):
    """This type of PV monitors one or more PVs using channal access and does a callback
    when one of the camonitors returns

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        monitored_pvs (list[PV] | list[str]): A list of PVs used to setup camonitoring.
        callbacks (list[Callable] | None): A list of functions to be called when the
            monitored PVs return. If none, then this PVs set function is called as the
            callback.

    Attributes:
        _monitor_list (list[tuple[PV, Callable]] | list[tuple[str, Callable]]): Used to
            keep track of which PVs we are monitoring and which functions the camonitor
            calls when they change value.
        _camonitor_handles (list[_Subscription]): Used to close camonitors if the a
            command is sent to pause monitoring.

    """

    def __init__(
        self,
        name: str,
        record_data: RecordData,
        monitored_pvs: list[PV] | list[str],
        callbacks: list[Callable] | None = None,
    ):
        super().__init__(name, record_data)
        self._monitor_list: list[tuple[PV | str, Callable]] = []
        self._camonitor_handles: list[_Subscription] = []
        if callbacks is None:
            callbacks = [self.set]
        self.monitor_pvs(monitored_pvs, callbacks)

    def monitor_pvs(self, pvs: list[PV] | list[str], callbacks: list[Callable]):
        """Setup camonitoring using the passed PVs and callbacks.

        Note: If len(callbacks) == 1 all pvs will use callbacks[0]. If len(callbacks) >1
            then pvs[i] will use callbacks[i] and len(callbacks) must equal len(pvs)

        Args:
            pvs (list[PV] | list[str]): A list of EPICS PVs to monitor using channal
                access.
            callbacks (list[Callable]): A list of functions to execute when the
                associated PV changes value.
        """
        pv_names: list[str] = []
        with_pv_as_str = False
        if len(callbacks) > 1:
            assert len(pvs) == len(callbacks)

        if isinstance(pvs[0], PV) or issubclass(type(pvs[0]), PV):
            pv_names = [pv.name for pv in pvs]
        elif isinstance(pvs[0], str):
            pv_names = pvs
            with_pv_as_str = True
        else:
            raise (
                ValueError(
                    "Monitored PV list must be filled with either str or PV type"
                )
            )

        if len(callbacks) == 1:
            # Use the same callback for all pvs
            callback = callbacks[0]
            for pv in pvs:
                self._monitor_list.append((pv, callback))

            if with_pv_as_str:
                self._camonitor_handles.extend(camonitor(pv_names, callback))
            else:
                self._camonitor_handles.extend(camonitor(pv_names, callback))
        else:
            # Use the specified callback for each pv
            for pv, callback in zip(pvs, callbacks, strict=True):
                self._monitor_list.append((pv, callback))
                if with_pv_as_str:
                    self._camonitor_handles.extend(camonitor(pv, callback))
                else:
                    self._camonitor_handles.extend(camonitor(pv.name, callback))

    def toggle_monitoring(self, enable):
        """Used to switch off this PVs monitoring by closing camonitor subscriptions or
        to re-enable monitoring by re-creating the subscriptions.

          Args:
            enable (bool): If true, we start monitoring, if false we stop it.
        """
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

    def set(self, value: RecordValue, index: int | None = None):
        """Set a value to this PVs softioc record.

        For the MonitorPV, the set function is called when a camonitor returns, if we
        are monitoring a list of PVs then an index is passed to this function.

        Args:
            value (RecordValue): The value to set to the softioc record.
            index (int): An optional index for when a list of camonitors returns a value
                which specified which index in the list of PVs returned.
        """
        logging.debug(f"PV: {self.name} changed to: {value}")
        self._record.set(value)
        # self._record.set_field("PROC", 1)


class RefreshPV(MonitorPV):
    """This PV monitors another PV and when it updates, we set our _record to the
    returned value and then force a third PV to update (refresh).

    Note: In the current implementation of VIRTAC, this PV is used to monitor an
        external PV in the tune feedbacks IOC. We store the value from the monitored
        PV in our _record, we then force a third PV (OffsetPV) to update. When this
        third PV updates, it reads the value from our _record and uses it as an offset
        which it adds to its own value before setting the result to itself and the pytac
        lattice.

    TODO: This PV does a lot of work at the moment, possible candidate for refactoring
        or removal.

    Args:
        name (str): Used to set self.name
        monitored_pv (PV | str): A PV to monitor and trigger refreshing.
        record_to_refresh (PV): The PV to pass to _recrod_to_refresh
        pv_to_cannibalise (PV): We take relevant variables from this PV, after which it
            should be discarded. TODO: It would be better if we didnt have to
            cannibalise an existing PV and could just create a new one.

    Attributes:
        _record_to_refresh (PV): The PV to refresh.
    """

    def __init__(
        self, name, monitored_pv: PV | str, record_to_refresh: PV, pv_to_cannibalise: PV
    ):
        super().__init__(name, None, [monitored_pv], [self.refresh])
        self._record_to_refresh: PV = record_to_refresh
        self._record: RecordWrapper = pv_to_cannibalise.get_record()
        self._pytac_items, self._pytac_field = pv_to_cannibalise.get_pytac_data()

    def refresh(self, value: RecordValue, index: int | None = None):
        """Set the value returned from the monitored PV to this PVs _record and then
        force an update of _record_to_refresh.

        Args:
            value (RecordValue): Value returned from camonitor
        """
        logging.debug(
            f"RefreshPV: {self.name} setting its value to {value} and forcing "
            f"{self._record_to_refresh.name} to process "
        )
        self._record.set(value)
        # self._record.set_field("PROC", 1)
        self._record_to_refresh.get_record().set_field("PROC", 1)


class InversionPV(MonitorPV):
    """Used to invert records containing a boolean or array of booleans, ie swap true to
    false and false to true and then save the result in its own waveform _record.

    Note: This class can either invert a single waveform record or a list of ai
        records. If invert_pvs contains more than 1 PV, then we assume the latter.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        invert_pvs (list[PV]): A list of PVs to monitor and then invert when they
            change value.

    Attributes:
        _invert_pvs (list[PV]): Same as invert_pvs argument.
    """

    def __init__(self, name: str, record_data: RecordData, invert_pvs: list[PV]):
        super().__init__(name, record_data, invert_pvs, [self.invert])
        self._invert_pvs: list[PV] = invert_pvs

    def invert(self, value: int | None = None, index: int | None = None):
        """Triggers this PV to caget the boolean values of all of its _invert_pv(s) and
            then invert them and set the result to _record.

        Args:
            value (RecordValue | None): This is ignored
            index (int | None): This is ignored
        """
        logging.debug(f"InversionPV: {self.name} inverting data")
        if (len(self._invert_pvs)) == 1:
            # Invert a single waveform record
            value = numpy.asarray(value, dtype=bool)
            value = numpy.asarray(numpy.invert(value), dtype=int)
        elif (len(self._invert_pvs)) > 1:
            # Invert a list of ai records
            value = numpy.array(
                [record.get() for record in self._invert_pvs], dtype=bool
            )
            value = numpy.asarray(numpy.invert(value), dtype=int)
        else:
            raise Exception
        self._record.set(value)
        self._record.set_field("PROC", 1)


class SummationPV(MonitorPV):
    """Used to sum values from a list of PVs, with the result set to this PVs _record.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        summate_pvs (list[PV]): A list of PVs to monitor and then sum when they
            change value.

    Attributes:
        _summate_pvs (list[PV]): Same as summate_pvs arg.
    """

    def __init__(self, name, record_data: RecordData, summate_pvs: list[PV]):
        super().__init__(name, record_data, summate_pvs, [self.summate])
        self._summate_pvs = summate_pvs

    def summate(self, value: RecordValue | None = None, index: int | None = None):
        """Caget a list of PV values and set the result to self._record

        Args:
            value (RecordValue): This is ignored
            index (int): This is ignored
        """

        logging.debug(f"SummationPV: {self.name} summing data")
        # TODO: This could be done more efficiently. We dont actually need to get all of
        # pvs, only self._summate_pvs[index], then we could modify only that index in
        # our waveform record. This is true for CollateionPV and InversionPV too.
        value = sum([pv.get() for pv in self._summate_pvs])
        self._record.set(value)
        # self._record.set_field("PROC", 1)


class CollationPV(MonitorPV):
    """Used to collate values from a list of PVs into an array, with the result set to
    this PVs _record.

    Note: Typically all pvs in collate_pvs will change at the same time and this can
    happen every time the simulation updates. This can result in a lot of unnecessary
    work, so we limit the update rate to 5Hz.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        collate_pvs (list[PV]): A list of PVs to monitor and then collate when they
            change value.

    Attributes:
        _collate_pvs (list[PV]): Same as collate_pvs arg.
        _update_required (bool): Tracks whether an update of the collation record is
            waiting to process.
        _minimum_time_between_updates (float): Used to force a maximum update rate of
            5Hz.
    """

    def __init__(self, name: str, record_data: RecordData, collate_pvs: list[PV]):
        super().__init__(name, record_data, collate_pvs, [self._set_update_required])
        self._last_update_time: float = time.time()
        self._collate_pvs: list[PV] = collate_pvs
        self._update_required: bool = False
        self._minimum_time_between_updates: float = 0.2
        cothread.Spawn(self._periodic_update)

    def _periodic_update(self):
        """Limit the rate at which we collate to 1/self._minimum_time_between_updates"""
        while True:
            if self._update_required:
                self.collate()
            else:
                cothread.Sleep(self._minimum_time_between_updates)

    def _set_update_required(
        self, value: RecordValue | None = None, index: int | None = None
    ):
        """Callback function executed when this PVs monitored PVs change value, rather
        than immediately doing the collation, we set this flag with the collation being
        triggered by the self._periodic_update polling loop.

        Args:
            value (RecordValue | None): This is ignored
            index (int | None): This is ignored
        """
        self._update_required = True

    def collate(self):
        """Get the current value of every PV in self._collate_pvs, collate them into
        a numpy array and set the result to this record."""

        logging.debug(f"CollationPV: {self.name} collating data")
        if time.time() - self._last_update_time < self._minimum_time_between_updates:
            cothread.Sleep(time.time() - self._last_update_time)
        value = numpy.array([record.get() for record in self._collate_pvs])

        self._record.set(value)
        self._record.set_field("PROC", 1)
        self._last_update_time = time.time()
        self._update_required = False


class CaPV:
    """Uses channel access to get and set an EPICS PV to a value. This PV does not need
    a softioc record.

    Args:
        name (str): Used to set self.name

    Attributes:
        self.name (str): The name used to define this PV and the name of the EPICS
            record to caget/caput.
    """

    def __init__(self, name: str):
        self.name = name

    def get(self) -> RecordValue:
        """Caget a value from an EPICS PV."""

        value = caget(self.name)
        logging.debug(f"Cagetting from PV: {self.name}, value: {value}")
        return value

    def set(self, value) -> ca_nothing | list[ca_nothing]:
        """Caput a value to an EPICS PV.

        Args:
            value (number): The value to caput to the PV.
        """
        logging.debug(f"Caputting to PV: {self.name}, value: {value}")
        return caput(self.name, value)
