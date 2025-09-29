"""Contains the PV subclasses which wrap softioc records and provide the link between
the softioc records and the simulation."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

import numpy
import pytac
from cothread.catools import _Subscription, camonitor
from softioc import builder
from softioc.pythonSoftIoc import RecordWrapper

RecordValueType: TypeAlias = int | float | numpy.ndarray
PytacItemType: TypeAlias = pytac.lattice.EpicsLattice | pytac.element.Element
CallbackType: TypeAlias = Callable[[RecordValueType, int | None], None]


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


class BasePV:
    """Stores the attributes and methods which allow the VIRTAC to control an
    EPICS PV.

    Attributes:
        self.__record (softioc.pythonSoftIoc.RecordWrapper): This softioc record is the
            heart of the PV class, the main purpose of PV objects is to manage the
            setting and getting of these records.
    """

    def __init__(self, name: str, record_data: RecordData | None) -> None:
        """
        Args:
            name: Used to identify this PV and its softioc record.
            record_data: Dataclass used to create this PVs softioc record.
        """
        logging.debug(f"Creating PV: {name}")
        self.name: str = name
        self.__record: RecordWrapper | None = None
        if record_data is not None:
            self.create_softioc_record(record_data)

    @property
    def _record(self) -> RecordWrapper:
        if self.__record is None:
            raise AttributeError(
                "Softioc record must be created before use.", name=self.__record
            )

        return self.__record

    @_record.setter
    def _record(self, new_record: RecordWrapper) -> None:
        if self.__record is not None:
            raise AttributeError(
                f"A softioc record could not be created for PV: {new_record.name}."
                "That name is already in use!"
            )

        self.__record = new_record

    def _on_update(self, value: RecordValueType, name: str) -> None:
        """The callback function called when the softioc record updates.

        This function and any overrides need to be kept FAST as they can be called
        rapidly by CA clients.

        Args:
            value: The value that has just been set to the record.
            name: The name of the softioc record that has just been set to.
        """
        logging.debug("Read value %s on pv %s", value, name)

    def set_record_field(self, field: str, value: RecordValueType) -> None:
        """Set a field on this PVs softioc record.

        Args:
            field: The EPICS field to set on the softioc record
            value: The value to set to the EPICS field
        """
        self._record.set_field(field, value)

    def create_softioc_record(
        self,
        record_data: RecordData,
    ) -> None:
        """Create this PVs softioc record.

        Args:
            record_data: Dataclass used to create this PVs softioc record.
        """
        logging.debug(f"Creating softioc record {self.name}")

        if self.__record is not None:
            raise AttributeError(
                f"A softioc record could not be created for PV: {self.name}."
                "That name is already in use!"
            )

        if record_data.record_type == RecordTypes.AI:
            self._record = builder.aIn(
                self.name,
                PREC=record_data.precision,
                HOPR=record_data.upper,
                LOPR=record_data.lower,
                SCAN=record_data.scan,
                initial_value=record_data.initial_value,
            )
        elif record_data.record_type == RecordTypes.AO:
            self._record = builder.aOut(
                self.name,
                PREC=record_data.precision,
                HOPR=record_data.upper,
                LOPR=record_data.lower,
                DRVH=record_data.drive_high,
                DRVL=record_data.drive_low,
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
                f"SoftIOC record type: {record_data.record_type} not supported. "
                f"Use one of the following: {[rt.value for rt in RecordTypes]}"
            )

    def get_record(self) -> RecordWrapper:
        """Return this PVs softioc record, care should be taken when manipulating the
        returned record.

        Returns:
            The softioc record object.
        """
        return self._record

    def get(self) -> RecordValueType:
        """Get the value stored in this PVs softioc record.

        Returns:
            The softioc record's value.
        """
        return self._record.get()

    def set(self, value: RecordValueType) -> None:
        """Set a value to this PVs softioc record.

        Args:
            value: The value to set to the softioc record.
        """
        logging.debug(f"PV: {self.name} changed to: {value}")
        self._record.set(value)


class ReadSimPV(BasePV):
    """This PV is used to read a value from the simulation using the pytac lattice and
    then set it to a softioc record.
    """

    def __init__(
        self,
        name: str,
        record_data: RecordData,
        pytac_items: list[PytacItemType],
        field: str,
    ) -> None:
        """
        Args:
            name: Used to identify this PV and its softioc record.
            record_data: Dataclass used to create this PVs softioc record.
            pytac_items:  A list of pytac elements or the pytac lattice itself which
                should be linked to this PV.
            pytac_field: The field on the pytac item(s) to set/get.
        """
        super().__init__(name, record_data)
        self._pytac_items: list[PytacItemType] = pytac_items
        self._pytac_field: str = field

    def append_pytac_item(self, pytac_item: PytacItemType) -> None:
        """Append a pytac item to the list of pytac items defined for this PV

        Args:
            pytac_item: The pytac element or lattice to append.
        """
        self._pytac_items.append(pytac_item)

    def update_from_sim(self) -> None:
        """Read a value from the simulation and set it to this PVs softioc record."""
        logging.debug(f"Updating pv {self.name}")
        try:
            value = cast(
                RecordValueType,
                self._pytac_items[0].get_value(
                    self._pytac_field, units=pytac.ENG, data_source=pytac.SIM
                ),
            )
            self.set(value)
        except pytac.exceptions.FieldException as e:
            logging.exception("PV is missing an expected pytac field")
            raise (e)


class ReadWriteSimPV(ReadSimPV):
    """This PV is used to write a value to the simulation from a softioc record using
    the pytac lattice.

    These PVs always have an associated readback PV, we do not simulate how the hardware
    would ramp up this readback PV, instead we simply set it equal to the value set to
    this PV.
    """

    def __init__(
        self,
        name: str,
        record_data: RecordData,
        read_pv: ReadSimPV,
        pytac_items: list[PytacItemType],
        pytac_field: str,
        offset_pv: BasePV | None = None,
    ) -> None:
        """
        Args:
            name: Used to identify this PV and its softioc record.
            record_data: Dataclass used to create this PVs softioc record.
            read_pv: The readback PV linked to this PV which reads from the lattice.
            pytac_items:  A list of pytac elements or the pytac lattice itself which
                should be linked to this PV.
            pytac_field: The field on the pytac item(s) to set/get.
            offset_pv: An optional PV which can be used to get an offset value which is
                appended to this pvs pytac item(s) when writing.
        """
        super().__init__(name, record_data, pytac_items, pytac_field)
        self._read_pv = read_pv
        self._offset_record: BasePV | None = offset_pv

    def _on_update(self, value: RecordValueType, name: str) -> None:
        """This function sets the passed value to self._pv_to_update._record by calling
        its set method. The set also sets value (with an additional offset from
        self._offset_pv) to the pytac item and field configured for self._pv_to_update.

        Args:
            value: The value that has just been set to self._record.
            name: The name of self._record object.
        """
        logging.debug("Read value %s on pv %s", value, name)
        if self._offset_record is not None:
            offset = self._offset_record.get()
            self.set(value, offset)
        else:
            self.set(value, None)

    def set(
        self, value: RecordValueType, offset: RecordValueType | None = None
    ) -> None:
        """Set a value to this PVs softioc record, update its pytac element(s)
            with the same value and then set the value to its read pv.

        Args:
            value: The value to set to the softioc record.
            offset: An optional offset value to add to this PVs pytac element but NOT to
                its softioc record.
        """
        logging.debug(f"PV: {self.name} changed to: {value}")
        if offset is not None:
            logging.debug("Adding offset of: %s new value is: %s", offset, value)
            value += offset

        # Some PVs such as the bend magnet PV have multiple pytac elements which
        # are all updated from the same PV value.

        # TODO: This could be a target for future improvement by supporting pairing
        # a single pytac element to multiple pyAT lattice elements.

        for item in self._pytac_items:
            logging.debug(
                "Updating field %s on lattice element %s for pv: %s to val: %s",
                self._pytac_field,
                item,
                self.name,
                value,
            )
            item.set_value(
                self._pytac_field,
                value,
                units=pytac.ENG,
                data_source=pytac.SIM,
            )

        # We set our new value to the _read_pv directly, rather than triggering
        # the _read_pv to read the updated value from the simulation. This is
        # faster and gives the same result as we do not simulate hardware ramping.
        self._read_pv.set(value)

    def attach_offset_record(self, offset_pv: BasePV) -> None:
        """Used to configure this PV with an offset PV in situations where the offset
        was created after this PV.

        Args:
            offset_pv: The PV object to be used during this PVs' records' on_update
                function.
        """
        logging.debug(f"Attaching offset record: {offset_pv} to PV: {self.name}")
        self._offset_record = offset_pv


class MonitorPV(BasePV):
    """This type of PV monitors one or more PVs using channal access and does a callback
    when one of the camonitors returns

    Attributes:
        _monitor_data ((list[tuple[list[str], list[CallbackType]]])): Used to keep track
            of which PVs we are monitoring and which functions the camonitor calls when
            they change value.
        _camonitor_handles (list[_Subscription]): Used to close camonitors if a
            command is sent to pause monitoring.
    """

    def __init__(
        self,
        name: str,
        record_data: RecordData | None,
        monitored_pv_names: list[str],
        callbacks: list[CallbackType] | None = None,
    ) -> None:
        """
        Args:
            name: Used to set self.name
            record_data: Dataclass used to create this PVs softioc record.
            monitored_pvs: A list of PV names used to setup camonitoring.
            callbacks: A list of functions to be called when the monitored PVs return.
                If none, then this PVs set function is called as the callback.
        """
        super().__init__(name, record_data)
        self._monitor_data: list[tuple[list[str], list[CallbackType]]] = []
        self._camonitor_handles: list[_Subscription] = []
        self._setup_pv_monitoring(monitored_pv_names, callbacks)

    def _setup_pv_monitoring(
        self,
        pv_names: list[str],
        callbacks: list[CallbackType] | None,
    ) -> None:
        """Setup camonitoring using the passed PV names and callbacks.

        If len(callbacks)>1 then a camonitor is created for each pv_name, callback pair.
        i.e pv_names[i] will trigger callbacks[i] on update.

        If len(callbacks)==1 then a single camonitor is created for all pv_names which
        will call the callback function with an additional index to identify which pv
        changed value.

        Args:
            pv_names: A list of PV names to monitor using channel access.
            callbacks: A list of functions to execute when the associated PV changes
                value.
        """
        if callbacks is None:
            callbacks = [self._callback]

        for pv_name in pv_names:
            for dataset in self._monitor_data:
                if pv_name in dataset[0]:
                    logging.warning(
                        f"The provided PV name: {pv_name} is already being "
                        "monitored. It is not recommended to setup multiple "
                        "camonitors for a single PV."
                    )

        if len(callbacks) == 1:
            self._setup_pv_monitoring_group(pv_names, callbacks)
        else:
            self._setup_pv_monitoring_individual(pv_names, callbacks)

    def _setup_pv_monitoring_group(
        self, pv_names: list[str], callback: list[CallbackType]
    ) -> None:
        self._monitor_data.append((pv_names, callback))
        self._camonitor_handles.extend(camonitor(pv_names, callback[0]))

    def _setup_pv_monitoring_individual(
        self, pv_names: list[str], callbacks: list[CallbackType]
    ) -> None:
        for pv_name, callback in zip(pv_names, callbacks, strict=True):
            self._monitor_data.append(([pv_name], [callback]))
            self._camonitor_handles.append(camonitor(pv_name, callback))

    def enable_monitoring(self) -> None:
        """Used to re-enable monitoring of this PV by re-creating the subscriptions."""
        logging.debug(f"Enabling monitoring for PV {self.name}")
        # We create a copy of monitor data, as set_pv_monitoring can append to this
        # original.
        monitor_data = self._monitor_data.copy()
        self._monitor_data.clear()
        for pv_list, callback in monitor_data:
            self._setup_pv_monitoring(pv_list, callback)

    def disable_monitoring(self) -> None:
        """Used to switch off this PVs monitoring by closing camonitor subscriptions."""
        logging.debug(f"Disabling monitoring for PV {self.name}")
        for handle in self._camonitor_handles:
            handle.close()
        self._camonitor_handles.clear()

    def _callback(self, value: RecordValueType, index: int | None = None) -> None:
        """Set a value to this PVs softioc record.

        For the MonitorPV, the set function is called when a camonitor returns, if we
        are monitoring a list of PVs then an index is passed to this function.
        """
        logging.debug(f"PV: {self.name} changed to: {value}")
        self.set(value)


class RefreshPV(MonitorPV):
    """This PV monitors another PV and when it updates, we set our _record to the
    returned value and then force a third PV to update (refresh).

    .. note::
        In the current implementation of VIRTAC, this PV is used to monitor an
        external PV in the tune feedbacks IOC. We store the value from the monitored
        PV in our _record, we then force a third PV (OffsetPV) to update. When this
        third PV updates, it reads the value from our _record and uses it as an offset
        which it adds to its own value before setting the result to itself and the pytac
        lattice.

    TODO: This PV does a lot of work at the moment, possible candidate for refactoring
        or removal. Removal would also allow us to guarentee that _record is not None.

    Attributes:
        _record_to_refresh (PV): The PV to refresh.
    """

    def __init__(
        self,
        name: str,
        monitored_pv_name: str,
        record_to_refresh: BasePV,
        pv_to_cannibalise: BasePV,
    ) -> None:
        """
        Args:
            name: Used to set self.name
            monitored_pv_name: A PV to monitor and trigger refreshing.
            record_to_refresh: The PV to pass to _record_to_refresh
            pv_to_cannibalise: We take relevant variables from this PV, after which it
                should be discarded.

        TODO: It would be better if we didnt have to.
            cannibalise an existing PV and could just create a new one.
        """
        super().__init__(name, None, [monitored_pv_name], [self._callback])
        self._record_to_refresh: BasePV = record_to_refresh
        self._record: RecordWrapper = pv_to_cannibalise.get_record()

    def _callback(self, value: RecordValueType, index: int | None = None) -> None:
        """Set the value returned from the monitored PV to this PVs _record and then
        force an update of _record_to_refresh.
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

    .. note:: This class can either invert a single waveform record or a list of ai
        records. If invert_pvs contains more than 1 PV, then we assume the latter.

    .. note:: In the current implementation of VIRTAC, this PV is being used to invert a
        list of SR01C-DI-EBPM-01:CF:ENABLED_S PVs, each containing a boolean value into
        a single waveform.
    """

    def __init__(
        self, name: str, record_data: RecordData, invert_pvs: list[BasePV]
    ) -> None:
        """
        Args:
            name: Used to set self.name
            record_data: Dataclass used to create this PVs softioc record.
            invert_pvs: A list of PVs to monitor and then invert when they change value.
        """
        if (len(invert_pvs)) == 0:
            raise AttributeError("InversionPV was not provided with any PVs to invert")
        super().__init__(
            name, record_data, [pv.name for pv in invert_pvs], [self._callback]
        )
        self._invert_pvs: list[BasePV] = invert_pvs

    def _callback(self, value: RecordValueType, index: int | None = None) -> None:
        """Triggers this PV to caget the boolean values of all of its _invert_pv(s) and
        then invert them and set the result to _record.
        """
        if index is None:
            # Invert a single waveform record
            record_data = numpy.asarray(value, dtype=bool)
            record_data = numpy.asarray(numpy.invert(record_data), dtype=int)
        else:
            # Invert the single element which changed
            record_data = numpy.copy(self._record.get())
            record_data[index] = not value

        self._record.set(record_data)
        logging.debug(
            f"InversionPV: {self.name} inverting data. New data: {record_data}"
        )


class SummationPV(MonitorPV):
    """
    Used to sum values from a list of PVs, with the result set to this PVs _record.
    """

    def __init__(
        self, name: str, record_data: RecordData, summate_pvs: list[BasePV]
    ) -> None:
        """
        Args:
            name: Used to set self.name
            record_data: Dataclass used to create this PVs softioc record.
            summate_pvs: A list of PVs to monitor and then sum when they change value.
        """
        super().__init__(
            name, record_data, [pv.name for pv in summate_pvs], [self._callback]
        )
        self._summate_pvs: list[BasePV] = summate_pvs

    def _callback(
        self, value: RecordValueType | None = None, index: int | None = None
    ) -> None:
        value = sum([pv.get() for pv in self._summate_pvs])
        self._record.set(value)
        logging.debug(f"SummationPV: {self.name} summing data. New value: {value}")


class CollationPV(MonitorPV):
    """Used to collate values from a list of PVs into an array, with the result set to
    this PVs _record.
    """

    def __init__(
        self, name: str, record_data: RecordData, collate_pvs: list[BasePV]
    ) -> None:
        """
        Args:
            name: Used to set self.name
            record_data: Dataclass used to create this PVs softioc record.
            collate_pv: A list of PVs to monitor and then collate when they change
                value.
        """
        super().__init__(
            name,
            record_data,
            [pv.name for pv in collate_pvs],
            [self._callback],
        )
        self._collate_pvs: list[BasePV] = collate_pvs

    def _callback(self, value: RecordValueType, index: int | None = None) -> None:
        if index is None:
            record_data = value
        else:
            record_data = numpy.copy(self._record.get())
            record_data[index] = value
        self._record.set(record_data)
        # logging.debug(
        #     f"CollationPV: {self.name} collating data. New value: {value}"
        # )
