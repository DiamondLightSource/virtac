"""This script does cagets from the live machine to get control limits and then uses
this data in combination with hardcoded data in this file to generate csv files. These
are used by the Virtac to create softioc records. This script should be manually run
using CA port 5064 whenever the csv files need updating.
"""

import argparse
import csv
import os
import sys
from typing import cast

import atip
import cothread
import numpy
import pytac
from cothread.catools import FORMAT_CTRL, caget

# Type alias for data to be stored in VIRTAC csv files
CSVData = list[tuple[str | int, ...]]

D2_RING_MODES = ["48"]


def generate_feedback_pvs(all_elements, lattice: pytac.lattice.EpicsLattice) -> CSVData:
    """Get feedback pvs. Also get families for tune feedback.

    Args:
        all_elements: a list of elements
        lattice: The pytac lattice being used by the virtual machine.
    Returns:
        Data to be written to csv.
    """

    if lattice.name in D2_RING_MODES:
        tune_quad_elements = set(
            all_elements.q0l
            + all_elements.q1l
            + all_elements.q2l
            + all_elements.q1n
            + all_elements.q2n
        )
    else:
        tune_quad_elements = set(
            all_elements.q1d
            + all_elements.q2d
            + all_elements.q3d
            + all_elements.q3b
            + all_elements.q2b
            + all_elements.q1b
        )

    # Data to be written is stored as a list of tuples each with structure:
    #     element index (int), field (str), pv (str), value (int), record_type (str).
    # We have special cases for four lattice fields that feedback systems read from.
    data: CSVData = [
        ("index", "field", "pv", "value", "record_type"),
        (0, "beam_current", "SR-DI-DCCT-01:SIGNAL", "300", "ai"),
        (0, "feedback_status", "CS-CS-MSTAT-01:FBSTAT", "2", "ai"),
        (0, "fofb_status", "SR01A-CS-FOFB-01:RUN", "0", "ao"),
        (0, "feedback_heart", "CS-CS-MSTAT-01:FBHEART", "10", "ao"),
    ]
    # Iterate over our elements to get the PV names.
    for elem in all_elements.hstr:
        pv_stem = elem.get_device("x_kick").name
        data.append((elem.index, "error_sum", pv_stem + ":ERCSUM", "0", "ai"))
        data.append((elem.index, "state", pv_stem + ":STATE", "2", "ai"))
    for elem in all_elements.vstr:
        pv_stem = elem.get_device("y_kick").name
        data.append((elem.index, "error_sum", pv_stem + ":ERCSUM", "0", "ai"))
        data.append((elem.index, "state", pv_stem + ":STATE", "2", "ai"))
    for elem in all_elements.bpm:
        data.append(
            (elem.index, "enabled", elem.get_pv_name("enabled", pytac.RB), "1", "ai")
        )
    # Add elements for Tune Feedback
    for elem in tune_quad_elements:
        data.append(
            (elem.index, "offset", elem.get_device("b1").name + ":OFFSET1", "0", "ai")
        )

    # BPM ID for the x axis of beam position plot
    bpm_ids = "["
    for pv in lattice.get_element_pv_names("BPM", "x", pytac.RB):
        bpm_ids += f"{int(pv[2:4]) + 0.1 * int(pv[14:16]):.1f} "
    bpm_ids = bpm_ids.rstrip()
    bpm_ids += "]"  # Should look like [1.1 1.2 1.3 ...]
    data.append(
        (
            0,
            "bpm_id",
            "SR-DI-EBPM-01:BPMID",
            bpm_ids,
            "wfmi",
        )
    )

    return data


def generate_bba_pvs(all_elements, symmetry: int) -> CSVData:
    """Data to be written is stored as a list of tuples each with structure:
    element index (int), field (str), pv (str), value (int), record_type (str).

    Args:
        all_elements: a list of elements
        symmetry: The number of cells in the lattice.
    Returns:
        Data to be written to csv.
    """
    data: CSVData = [("index", "field", "pv", "value", "record_type")]
    pv_stem: str = ""
    # Iterate over the BPMs to construct the PV names.
    for elem in all_elements.bpm:
        pv_stem = elem.get_device("enabled").name
        data.append(
            (elem.index, "golden_offset_x", pv_stem + ":CF:GOLDEN_X_S", "0", "ao")
        )
        data.append(
            (elem.index, "golden_offset_y", pv_stem + ":CF:GOLDEN_Y_S", "0", "ao")
        )
        data.append((elem.index, "bcd_offset_x", pv_stem + ":CF:BCD_X_S", "0", "ao"))
        data.append((elem.index, "bcd_offset_y", pv_stem + ":CF:BCD_Y_S", "0", "ao"))
        data.append((elem.index, "bba_offset_x", pv_stem + ":CF:BBA_X_S", "0", "ao"))
        data.append((elem.index, "bba_offset_y", pv_stem + ":CF:BBA_Y_S", "0", "ao"))
    for cell in range(1, symmetry + 1):
        padded_cell: str = str(cell).zfill(2)
        pv_stem = f"SR{padded_cell}A-CS-FOFB-01"
        # Waveform records
        data.append(
            (
                cell,
                f"cell_{padded_cell}_excite_start_times",
                f"{pv_stem}:EXCITE:START_TIMES",
                str(numpy.zeros(18)),
                "wfmo",
            )
        )
        data.append(
            (
                cell,
                f"cell_{padded_cell}_excite_amps",
                f"{pv_stem}:EXCITE:AMPS",
                str(numpy.zeros(18)),
                "wfmo",
            )
        )
        data.append(
            (
                cell,
                f"cell_{padded_cell}_excite_deltas",
                f"{pv_stem}:EXCITE:DELTAS",
                str(numpy.zeros(18)),
                "wfmo",
            )
        )
        data.append(
            (
                cell,
                f"cell_{padded_cell}_excite_ticks",
                f"{pv_stem}:EXCITE:TICKS",
                str(numpy.zeros(18)),
                "wfmo",
            )
        )
        # ao record
        data.append(
            (
                cell,
                f"cell_{padded_cell}_excite_prime",
                f"{pv_stem}:EXCITE:PRIME",
                "0",
                "ao",
            )
        )
    return data


def get_element_pv_data(
    pytac_item: pytac.lattice.Lattice | pytac.element.Element,
    pvs: list[str],
    data: CSVData,
) -> None:
    """Get the control limits and precision values from the live machine for
    all normal PVS.

    Args:
        pytac_item: An element of the pytac lattice or the lattice itself
        pvs: A list of pv names which we have already found
        data: A list of tuples, with each tuple being a collection of data about one pv.
    """
    field_data: dict = pytac_item.get_fields()
    lat_fields: set[str] = set(field_data[pytac.LIVE]).intersection(
        set(field_data[pytac.SIM])
    )
    # These pvs need to be configured with their SCAN fields set to .1 second. This is
    # different to the SCAN field in the LIVE pv, so we cant just caget it.
    scan_pvs: list[str] = ["SR-DI-EMIT-01:HEMIT", "SR-DI-EMIT-01:VEMIT"]
    for field in lat_fields:
        if not isinstance(pytac_item.get_device(field), pytac.device.SimpleDevice):
            rb_pv: str = pytac_item.get_pv_name(field, pytac.RB)
            if rb_pv not in pvs:
                ctrl = caget(rb_pv, format=FORMAT_CTRL, timeout=10)
                pvs.append(rb_pv)
                data.append(
                    (
                        rb_pv,
                        ctrl.upper_ctrl_limit,
                        ctrl.lower_ctrl_limit,
                        ctrl.precision,
                        ctrl.upper_disp_limit,
                        ctrl.lower_disp_limit,
                        ".1 second" if rb_pv in scan_pvs else "I/O Intr",
                    )
                )
                try:
                    sp_pv: str = pytac_item.get_pv_name(field, pytac.SP)
                except pytac.exceptions.HandleException:
                    pass
                else:
                    if sp_pv not in pvs:
                        ctrl = caget(sp_pv, format=FORMAT_CTRL, timeout=10)
                        data.append(
                            (
                                sp_pv,
                                ctrl.upper_ctrl_limit,
                                ctrl.lower_ctrl_limit,
                                ctrl.precision,
                                ctrl.upper_disp_limit,
                                ctrl.lower_disp_limit,
                                ".1 second" if sp_pv in scan_pvs else "Passive",
                            )
                        )


def generate_pv_limits(lattice: pytac.lattice.Lattice) -> CSVData:
    """Loop through each element in the lattice and spawn a cothread which will then
    do a caget to get pv data for the element.

    Args:
        lattice: The pytac lattice being used by the virtual machine.

    Returns:
        Data to be written to csv.
    """
    data: CSVData = [
        ("pv", "upper", "lower", "precision", "drive_high", "drive_low", "scan")
    ]
    pvs: list[str] = []
    caget_handles: list[cothread.Spawn] = []
    # Add limits for PVs connected to lattice element and the lattice itself
    pytac_items: list[pytac.lattice.Lattice | pytac.element.Element] = list(lattice)
    pytac_items.insert(0, lattice)
    for item in pytac_items:
        caget_handles.append(cothread.Spawn(get_element_pv_data, item, pvs, data))
    for caget_handle in caget_handles:
        caget_handle.Wait()
    return data


def generate_mirrored_pvs(lattice: pytac.lattice.Lattice) -> CSVData:
    """Structure of data:

    output_type:
        The type of output record to create, only 'ai', 'longIn',
        'Waveform' types are currently supported; if '' then output to an
        existing in record already created in VirtacServer, 'caput' is also a
        special case it creates a mask for cothread.catools.caput calling
        set(value) on this mask will call caput with the output PV and the
        passed value.

    mirror_type (The type of mirroring to apply):
        - basic: set the value of the input record to the output record.
        - summate: sum the values of the input records and set the result to
            the output record.
        - collate: create a Waveform record from the values of the input PVs.
        - transform: apply the specified transformation function to the value
            of the input record and set the result to the output record. N.B.
            the only transformation type currently supported is 'inverse'.

    in_pv:
        The PV(s) to be monitored, on change mirror is updated, if multiple
        then the PVs should be separated by a comma and one space.

    out_pv:
        The single PV to output to, if a 'record type' is spcified then a new
        record will be created and so must not exist already.

    value:
        The inital value of the output record.

    refresh:
        Whether the out_pv should have its softioc record's SCAN field set to
        '.1 second' which will cause it to process every second.

    Args:
        lattice: The pytac lattice being used by the virtual machine.
    Returns:
        Data to be written to csv.
    """
    data: CSVData = [("output_type", "mirror_type", "in_pv", "out_pv", "value", "scan")]
    # Tune PV aliases.
    tune: list[str] = [
        lattice.get_value("tune_x", pytac.RB, data_source=pytac.SIM),
        lattice.get_value("tune_y", pytac.RB, data_source=pytac.SIM),
    ]
    data.append(
        (
            "ai",
            "basic",
            "SR23C-DI-TMBF-01:X:TUNE:TUNE",
            "SR23C-DI-TMBF-01:TUNE:TUNE",
            tune[0],
            ".1 second",
        )
    )
    data.append(
        (
            "ai",
            "basic",
            "SR23C-DI-TMBF-01:Y:TUNE:TUNE",
            "SR23C-DI-TMBF-02:TUNE:TUNE",
            tune[1],
            ".1 second",
        )
    )
    # Combined emittance and average emittance PVs.
    emit: list[float] = [
        lattice.get_value("emittance_x", pytac.RB, data_source=pytac.SIM),
        lattice.get_value("emittance_y", pytac.RB, data_source=pytac.SIM),
    ]
    data.append(
        (
            "ai",
            "basic",
            "SR-DI-EMIT-01:HEMIT",
            "SR-DI-EMIT-01:HEMIT_MEAN",
            str(emit[0]),
            "I/O Intr",
        )
    )
    data.append(
        (
            "ai",
            "basic",
            "SR-DI-EMIT-01:VEMIT",
            "SR-DI-EMIT-01:VEMIT_MEAN",
            str(emit[1]),
            "I/O Intr",
        )
    )
    data.append(
        (
            "ai",
            "summate",
            "SR-DI-EMIT-01:HEMIT, SR-DI-EMIT-01:VEMIT",
            "SR-DI-EMIT-01:EMITTANCE",
            str(sum(emit)),
            "I/O Intr",
        )
    )
    # Electron BPMs enabled.
    bpm_enabled_pvs = cast(
        list[str], lattice.get_element_pv_names("BPM", "enabled", pytac.RB)
    )
    data.append(
        (
            "wfmi",
            "inverse",
            ", ".join(bpm_enabled_pvs),
            "SR-DI-EBPM-01:ENABLED",
            str(numpy.zeros(len(bpm_enabled_pvs))),
            "I/O Intr",
        )
    )
    # BPM x positions for display on diagnostics screen.
    bpm_x_pvs = cast(list[str], lattice.get_element_pv_names("BPM", "x", pytac.RB))
    data.append(
        (
            "wfmi",
            "collate",
            ", ".join(bpm_x_pvs),
            "SR-DI-EBPM-01:SA:X",
            str(numpy.zeros(len(bpm_x_pvs))),
            "I/O Intr",
        )
    )
    # BPM y positions for display on diagnostics screen.
    bpm_y_pvs = cast(list[str], lattice.get_element_pv_names("BPM", "y", pytac.RB))
    data.append(
        (
            "wfmi",
            "collate",
            ", ".join(bpm_y_pvs),
            "SR-DI-EBPM-01:SA:Y",
            str(numpy.zeros(len(bpm_y_pvs))),
            "I/O Intr",
        )
    )
    return data


def generate_tune_pvs(lattice: pytac.lattice.Lattice) -> CSVData:
    """Get the PVs associated with the tune feedback system, the structure of
    data is:

    set_pv:
        The PV to set the offset to.

    offset_pv:
        The PV which the set pv reads the offset from.

    delta_pv:
        The PV to get the offset from.

    Args:
        lattice: The pytac lattice being used by the virtual machine.
    Returns:
        Data to be written to csv.
    """
    data: CSVData = [("set_pv", "offset_pv", "delta_pv")]
    # Offset PV for quadrupoles in tune feedback.
    tune_pvs: list[str] = []
    offset_pvs: list[str] = []
    delta_pvs: list[str] = []

    if lattice.name in D2_RING_MODES:
        tune_quad_families = ["Q0L", "Q1L", "Q2L", "Q1N", "Q2N"]
    else:
        tune_quad_families = ["Q1D", "Q2D", "Q3D", "Q3B", "Q2B", "Q1B"]

    for family in tune_quad_families:
        tune_pvs.extend(lattice.get_element_pv_names(family, "b1", pytac.SP))
    for pv in tune_pvs:
        offset_pvs.append(":".join([pv.split(":")[0], "OFFSET1"]))
        delta_pvs.append(f"SR-CS-TFB-01:{pv[2:4]}{pv[9:12]}{pv[13:15]}:I")
    for tune_pv, offset_pv, delta_pv in zip(
        tune_pvs, offset_pvs, delta_pvs, strict=False
    ):
        data.append((tune_pv, offset_pv, delta_pv))
    return data


def write_data_to_file(data: CSVData, filename: str, ring_mode: str) -> None:
    """Write the collected data to a .csv file with the given name. If the file
    already exists it will be overwritten.

    Args:
        data: A list of tuples, the data to write to the .csv file.
        filename: The name of the .csv file to write the data to.
    """
    if not filename.endswith(".csv"):
        filename += ".csv"
    filepath = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "data", ring_mode, filename
    )
    column_titles = data[0]
    sorted_data = sorted(data[1:])
    with open(filepath, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows([column_titles] + sorted_data)


def parse_arguments() -> argparse.Namespace:
    """The arguments passed to this script to configure how the csv is to be created"""
    parser = argparse.ArgumentParser(
        description="Generate CSV file to define the PVs served by the "
        "virtual accelerator IOC."
    )
    parser.add_argument(
        "ring_mode",
        nargs="?",
        type=str,
        help="Ring mode name",
        default="I04",
    )
    parser.add_argument(
        "--offline",
        help="Generate csv files without gettiing limits data from the live machine",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--feedback",
        help="Filename for output feedback PVs CSV file",
        default="feedback.csv",
    )
    parser.add_argument(
        "--limits",
        help="Filename for output PV limits CSV file",
        default="limits.csv",
    )
    parser.add_argument(
        "--mirrored",
        help="Filename for output mirrored PVs CSV file",
        default="mirrored.csv",
    )
    parser.add_argument(
        "--tune",
        help="Filename for output tune feedback offset PVs CSV file",
        default="tunefb.csv",
    )
    parser.add_argument(
        "--bba",
        help="Filename for output beam-based-alignment PVs CSV file",
        default="bba.csv",
    )
    return parser.parse_args()


def main():
    # Set the default string printing options for numpy arrays so that they are properly
    # formatted when outputting them to the csv file
    with numpy.printoptions(threshold=sys.maxsize, linewidth=100000):
        args = parse_arguments()
        lattice = atip.utils.loader(args.ring_mode)
        # all_elements is a class with an attribute for each element family, where
        # that attribute is a list of all elements of that family.
        all_elements = atip.utils.preload(lattice)

        print("Creating feedback PVs CSV file.")
        data = generate_feedback_pvs(all_elements, lattice)
        write_data_to_file(data, args.feedback, args.ring_mode)

        print("Creating BBA PVs CSV file.")
        data = generate_bba_pvs(all_elements, cast(int, lattice.symmetry))
        write_data_to_file(data, args.bba, args.ring_mode)

        if not args.offline:
            print("Creating limits PVs CSV file.")
            data = generate_pv_limits(lattice)
            if len(data) <= 1:
                print("No limits data found, limits.csv will not be updated.")
            else:
                write_data_to_file(data, args.limits, args.ring_mode)

        print("Creating mirrored PVs CSV file.")
        data = generate_mirrored_pvs(lattice)
        write_data_to_file(data, args.mirrored, args.ring_mode)

        print("Creating tune PVs CSV file.")
        data = generate_tune_pvs(lattice)
        write_data_to_file(data, args.tune, args.ring_mode)


if __name__ == "__main__":
    main()
