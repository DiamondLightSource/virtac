import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy
import pytac
from cothread.catools import _Subscription, camonitor
from softioc import builder
from softioc.pythonSoftIoc import RecordWrapper

RecordValueType: TypeAlias = int | float | numpy.typing.NDArray
PytacItemType: TypeAlias = pytac.lattice.Lattice | pytac.element.Element


class RecordTypes(StrEnum):
    """Currently supported EPICS sofioc record types"""

    AI = "ai"
    AO = "ao"
    WAVEFORM_OUT = "wfmo"
    WAVEFORM_IN = "wfmi"
    MBBI = "mbbi"


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
    always_update: bool = False
    initial_value: RecordValueType = 0

    def __post_init__(self):
        if not isinstance(self.record_type, str):
            raise ValueError("Record field `record_type` must be of string type")
        if not isinstance(self.scan, str):
            raise ValueError("Record field `scan` must be of string type")
        if not isinstance(self.pini, str):
            raise ValueError("Record field `pini` must be of string type")


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
        self._pytac_items: list[PytacItemType] = []
        self._pytac_field: str = ""
        if record_data is not None:
            self.create_softioc_record(record_data)

    def _on_update(self, value: RecordValueType, name: str):
        """The callback function called when the softioc record updates.

        This functions needs to be kept FAST as it can be called rapidly by CA clients.

        Args:
            value (RecordValue): The value that has just been set to the record.
            name (str): The name of the softioc record that has just been set to.
        """
        logging.debug("Read value %s on pv %s", value, name)

    def get_pytac_data(self) -> tuple[list[PytacItemType], str]:
        """Return the list of pytac elements and the field defined for this PV"""
        return self._pytac_items, self._pytac_field

    def append_pytac_item(self, pytac_item: PytacItemType):
        """Append a pytac item to the list of pytac items defined for this PV

        Args:
            pytac_item (list[PytacItem]): The pytac element or lattice to append.
        """
        self._pytac_items.append(pytac_item)

    def set_pytac_field(self, field: str):
        """Set this PVs pytac field to the passed value

        Args:
            field (str): The pytac field to the value to.
        """
        self._pytac_field = field

    def set_record_field(self, field: str, value: RecordValueType | str):
        """Set a field on this PVs softioc record

        Args:
            field (softioc.field): The EPICS field to set on the softioc record
            value (RecordValueType | str): The value to set to the EPICS field"""
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
        if record_data.record_type == RecordTypes.AI:
            self._record = builder.aIn(
                self.name,
                PREC=record_data.precision,
                LOPR=record_data.lower,
                HOPR=record_data.upper,
                SCAN=record_data.scan,
                initial_value=record_data.initial_value,
            )
        elif record_data.record_type == RecordTypes.AO:
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
        elif record_data.record_type == RecordTypes.WAVEFORM_IN:
            self._record = builder.WaveformIn(
                self.name,
                initial_value=record_data.initial_value,
                PINI=record_data.pini,
                SCAN=record_data.scan,
            )
        elif record_data.record_type == RecordTypes.WAVEFORM_OUT:
            self._record = builder.WaveformOut(
                self.name,
                initial_value=record_data.initial_value,
                PINI=record_data.pini,
                SCAN=record_data.scan,
                always_update=record_data.always_update,
            )
        elif record_data.record_type == RecordTypes.MBBI:
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

        Care should be taken when manipulating the returned record.
        """
        return self._record

    def get(self) -> RecordValueType:
        """Get the value stored in this PVs softioc record"""
        return self._record.get()

    def set(self, value: RecordValueType, offset: RecordValueType | None = None):
        """Set a value to this PVs softioc record, and then update its pytac element(s)
            with the same value.

        Args:
            value (RecordValue): The value to set to the softioc record.
            offset (RecordValue): An offset value to add to this PVs pytac element but
                not to its softioc record.
        """

        logging.debug(f"PV: {self.name} changed to: {value}")
        self._record.set(value)
        if offset is not None:
            logging.debug("Adding offset of: %s new value is: %s", offset, value)
            value += offset

        pytac_items, field = self.get_pytac_data()
        # Some PVs such as the bend magnet PV have multiple pytac elements which
        # are updated from the same PV value.
        for item in pytac_items:
            logging.debug(
                "Updating field %s on lattice element %s for pv: %s to val: %s",
                field,
                item,
                self.name,
                value,
            )
            item.set_value(
                field,
                value,
                units=pytac.ENG,
                data_source=pytac.SIM,
            )


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

    def set(self, value: RecordValueType):
        """Set a value to this PVs softioc record.
        Args:
            value (RecordValue): The value to set to the softioc record.
        """
        logging.debug(f"PV: {self.name} changed to: {value}")
        self._record.set(value)


class SetpointPV(PV):
    """This PV is used to set a value to another PV and its associated pytac item(s).

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        in_pv (PVType): The PV which is to be updated when the SetpointPV's
            softioc record processes.
    """

    def __init__(self, name: str, record_data: RecordData, in_pv: PVType):
        super().__init__(name, record_data)
        self._in_pv: PVType = in_pv

    def _on_update(self, value: RecordValueType, name: str):
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


class OffsetPV(SetpointPV):
    """This function sets the passed value to self._in_pv._record by calling its set
        method. The set also sets value to the pytac item and field configured for
        self._in_pv.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        in_pv (PVType): The PV object to pass to _in_pv
        offset_pv (PVType | None): The PV object to pass to _offset_pv. It is optional
            to pass this at initialisation, but if not passed, it must be later attached
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
        in_pv: PVType,
        offset_pv: PVType | None = None,
    ):
        super().__init__(name, record_data, in_pv)
        self._offset_record: PVType | None = offset_pv

    def _on_update(self, value: RecordValueType, name: str):
        """This function sets the passed value to self._in_pv._record by calling its set
        method. The set also sets value (with an additional offset from self._offset_pv)
        to the pytac item and field configured for self._in_pv.

        This function is called whenever this PVs softioc record processes. It needs to
        be kept FAST as it can be called rapidly by CA clients which are writing to
        self._record.

        Args:
            value (RecordValue): The value that has just been set to self._record.
            name (str): The name of self._record object.
        """
        logging.debug("Read value %s on pv %s", value, name)
        if self._offset_record is not None:
            offset = self._offset_record.get()
        else:
            raise AttributeError(
                f"No offset record specified for OffsetPV: {self.name}"
            )
        self._in_pv.set(value, offset)

    def attach_offset_record(self, offset_pv: PVType):
        """Used to configure this PV with an offset PV in situations where the offset
        was created after this PV.

        Args:
            offset_pv (PV): The PV object to be used during this PVs' records' on_update
            function.
        """
        logging.debug(f"Attaching offset record: {offset_pv} to PV: {self.name}")
        self._offset_record = offset_pv


class MonitorPV(PV):
    """This type of PV monitors one or more PVs using channal access and does a callback
    when one of the camonitors returns

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        monitored_pvs (list[str]): A list of PV names used to setup camonitoring.
        callbacks (list[Callable] | None): A list of functions to be called when the
            monitored PVs return. If none, then this PVs set function is called as the
            callback.

    Attributes:
        _monitor_data ((list[tuple[list[str], list[Callable]]])): Used to keep track of
            which PVs we are monitoring and which functions the camonitor calls when
            they change value.
        _camonitor_handles (list[_Subscription]): Used to close camonitors if a
            command is sent to pause monitoring.
    """

    def __init__(
        self,
        name: str,
        record_data: RecordData | None,
        monitored_pv_names: list[str],
        callbacks: list[Callable] | None = None,
    ):
        super().__init__(name, record_data)
        self._monitor_data: list[tuple[list[str], list[Callable]]] = []
        self._camonitor_handles: list[_Subscription] = []
        self.setup_pv_monitoring(monitored_pv_names, callbacks)

    def setup_pv_monitoring(self, pv_names, callbacks):
        """Setup camonitoring using the passed PV names and callbacks.

        If len(callbacks)>1 then a camonitor is created for each pv_name, callback pair.
        i.e pv_names[i] will trigger callbacks[i] on update.

        If len(callbacks)==1 then a single camonitor is created for all pv_names which
        will call the callback function with an additional index to identify which pv
        changed value.

        Args:
            pv_names (list[str]): A list of PV names to monitor using channel access.
            callbacks (list[Callable]): A list of functions to execute when the
                associated PV changes value.
        """
        if callbacks is None:
            callbacks = [self.set]

        for pv_name in pv_names:
            if not isinstance(pv_name, str):
                raise TypeError(f"PV name must be a string, not {type(pv_name)}")
            elif pv_name in self._monitor_data:
                logging.warning(
                    f"The provided PV name: {pv_name} is already being monitored."
                )
                pv_names.remove(pv_name)

        if len(callbacks) == 1:
            self._setup_pv_monitoring_group(pv_names, callbacks)
        else:
            self._setup_pv_monitoring_individual(pv_names, callbacks)

    def _setup_pv_monitoring_individual(
        self, pv_names: list[str], callbacks: list[Callable]
    ):
        for pv_name, callback in zip(pv_names, callbacks, strict=True):
            self._monitor_data.append(([pv_name], [callback]))
            self._camonitor_handles.append(camonitor(pv_name, callback))

    def _setup_pv_monitoring_group(self, pv_names: list[str], callback: list[Callable]):
        self._monitor_data.append((pv_names, callback))
        self._camonitor_handles.extend(camonitor(pv_names, callback[0]))

    def enable_monitoring(self):
        """Used to re-enable monitoring of this PV by re-creating the subscriptions."""
        logging.debug(f"Enabling monitoring for PV {self.name}")
        # We create a copy of monitor data, as set_pv_monitoring can append to this
        # original.
        monitor_data = self._monitor_data.copy()
        for pv_list, callback in monitor_data:
            self.setup_pv_monitoring(pv_list, callback)

    def disable_monitoring(self):
        """Used to switch off this PVs monitoring by closing camonitor subscriptions."""
        logging.debug(f"Disabling monitoring for PV {self.name}")
        for handle in self._camonitor_handles:
            handle.close()
        self._camonitor_handles.clear()

    def set(self, value: RecordValueType, index: int | None = None):
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
        monitored_pv_name (str): A PV to monitor and trigger refreshing.
        record_to_refresh (PVType): The PV to pass to _record_to_refresh
        pv_to_cannibalise (PVType): We take relevant variables from this PV, after
            which it should be discarded. TODO: It would be better if we didnt have to.
            cannibalise an existing PV and could just create a new one.

    Attributes:
        _record_to_refresh (PV): The PV to refresh.
    """

    def __init__(
        self,
        name,
        monitored_pv_name: str,
        record_to_refresh: PVType,
        pv_to_cannibalise: PVType,
    ):
        super().__init__(name, None, [monitored_pv_name], [self.refresh])
        self._record_to_refresh: PVType = record_to_refresh
        self._record: RecordWrapper = pv_to_cannibalise.get_record()
        self._pytac_items, self._pytac_field = pv_to_cannibalise.get_pytac_data()

    def refresh(self, value: RecordValueType, index: int | None = None):
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
        self._record_to_refresh.set_record_field("PROC", 1)


class InversionPV(MonitorPV):
    """Used to invert records containing a boolean or array of booleans, ie swap true to
    false and false to true and then save the result in its own waveform _record.

    Note: This class can either invert a single waveform record or a list of ai
        records. If invert_pvs contains more than 1 PV, then we assume the latter.

    Note: In the current implementation of VIRTAC, this PV is being used to invert a
        list of SR01C-DI-EBPM-01:CF:ENABLED_S PVs, each containing a boolean value into
        a single waveform.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        invert_pvs (list[PVType]): A list of PVs to monitor and then invert when
            they change value.
    """

    def __init__(self, name: str, record_data: RecordData, invert_pvs: list[PVType]):
        super().__init__(
            name, record_data, [pv.name for pv in invert_pvs], [self.invert]
        )
        self._invert_pvs: list[PVType] = invert_pvs

    def invert(self, value: RecordValueType, index: int | None = None):
        """Triggers this PV to caget the boolean values of all of its _invert_pv(s) and
            then invert them and set the result to _record.

        Args:
            value (RecordValue): The value to invert and save to this PVs record
            index (int | None): This is ignored if only a single invert_pv is being
                monitored.
        """
        if (len(self._invert_pvs)) == 1:
            # Invert a single waveform record
            value = numpy.asarray(value, dtype=bool)
            value = numpy.asarray(numpy.invert(value), dtype=int)
        elif (len(self._invert_pvs)) > 1:
            # Invert the single element which changed
            record_data = numpy.copy(self._record.get())
            record_data[index] = not value
            self._record.set(record_data)
        elif (len(self._invert_pvs)) == 0:
            raise AttributeError("InversionPV was not provided with any PVs to invert")

        logging.debug(
            f"InversionPV: {self.name} inverting data. New data: {record_data}"
        )


class SummationPV(MonitorPV):
    """Used to sum values from a list of PVs, with the result set to this PVs _record.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        summate_pvs (list[PVType]): A list of PVs to monitor and then sum when they
            change value.
    """

    def __init__(self, name, record_data: RecordData, summate_pvs: list[PVType]):
        super().__init__(
            name, record_data, [pv.name for pv in summate_pvs], [self.summate]
        )
        self._summate_pvs: list[PVType] = summate_pvs

    def summate(self, value: RecordValueType | None = None, index: int | None = None):
        """Caget a list of PV values and set the result to self._record

        Args:
            value (RecordValue): This is ignored
            index (int): This is ignored
        """

        value = sum([pv.get() for pv in self._summate_pvs])
        self._record.set(value)
        logging.debug(f"SummationPV: {self.name} summing data. New value: {value}")


class CollationPV(MonitorPV):
    """Used to collate values from a list of PVs into an array, with the result set to
    this PVs _record.

    Args:
        name (str): Used to set self.name
        record_data (RecordData): Dataclass used to create this PVs softioc record.
        collate_pvs (list[PVType]): A list of PVs to monitor and then collate when they
            change value.
    """

    def __init__(self, name: str, record_data: RecordData, collate_pvs: list[PVType]):
        super().__init__(
            name,
            record_data,
            [pv.name for pv in collate_pvs],
            [self.collate],
        )
        self._collate_pvs: list[PVType] = collate_pvs

    def collate(self, value: RecordValueType, index: int):
        """Update this PVs waveform record using the given value and index"""
        record_data = numpy.copy(self._record.get())
        record_data[index] = value
        self._record.set(record_data)
        logging.debug(
            f"CollationPV: {self.name} collating data. New data: {record_data}"
        )
