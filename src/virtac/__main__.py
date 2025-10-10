import asyncio
import logging
import os
import socket
from argparse import ArgumentError, ArgumentParser
from pathlib import Path
from typing import cast
from warnings import warn

from aioca import CANothing, caget
from softioc import asyncio_dispatcher, builder, softioc

from virtac import virtac_server

__all__ = ["main"]

LOG_FORMAT = "%(asctime)s %(message)s"
DATADIR = Path(__file__).absolute().parent / "data"


def parse_arguments():
    """Parse command line arguments sent to virtac"""
    parser = ArgumentParser()
    parser.add_argument(
        "ring_mode",
        nargs="?",
        type=str,
        help="The ring mode to be used, e.g., IO4 or DIAD",
    )
    parser.add_argument(
        "-e",
        "--disable-emittance",
        help="Disable the simulator's time-consuming emittance calculation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--disable-chromaticity",
        help="Disable chromaticity calculations",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--disable-radiation",
        help="Disable radiation calculations in the simulation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--linopt-function",
        help="Which pyAT linear optics function to use: linopt2, linopt4, linopt6. "
        "Default is linopt6",
        default="linopt6",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--disable-tfb",
        help="Disable extra simulated hardware required by the Tune Feedback system",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Increase logging verbosity. Default is WARNING. -v=INFO -vv=DEBUG",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="__version__",
    )
    return parser.parse_args()


def check_sim_params(args):
    """Check that we have a valid combination of simulation parameters."""
    if args.disable_radiation:
        if args.linopt_function == "linopt6":
            raise ArgumentError(
                None,
                f"Cannot disable radiation when using linopt function: "
                f"{args.linopt_function}",
            )
        if not args.disable_emittance:
            raise ArgumentError(
                None,
                "You cannot calculate emittance with radiation disabled",
            )
    else:
        if args.linopt_function == "linopt2" or args.linopt_function == "linopt4":
            raise ArgumentError(
                None,
                "You must disable radiation to use linopt function: "
                f"{args.linopt_function}",
            )


def configure_ca():
    """Setup channel access settings for our CA server and for accessing PVs
    from other IOCs. We will be creating a python softioc IOC which automatically
    creates a CA server to serve our PVs.
    """

    # Warn if set to default EPICS port(s) as this will likely cause PV conflicts.
    conflict_warning = ", this may lead to conflicting PV names with production IOCs."
    epics_env_vars = [
        "EPICS_CA_REPEATER_PORT",
        "EPICS_CAS_SERVER_PORT",
        "EPICS_CA_SERVER_PORT",
        "EPICS_CAS_BEACON_PORT",
    ]
    ports_list = [int(os.environ.get(env_var, 0)) for env_var in epics_env_vars]
    if 5064 in ports_list or 5065 in ports_list:
        warn(
            f"At least one of {epics_env_vars} is set to 5064 or 5065"
            + conflict_warning,
            stacklevel=1,
        )
    elif all(port == 0 for port in ports_list):
        warn(
            "No EPICS port set, default base port (5064) will be used"
            + conflict_warning,
            stacklevel=1,
        )

    # Avoid PV conflict between multiple IP interfaces on the same machine.
    primary_ip = socket.gethostbyname(socket.getfqdn())
    if "EPICS_CAS_INTF_ADDR_LIST" in os.environ.keys():
        warn(
            "Pre-existing 'EPICS_CAS_INTF_ADDR_LIST' value" + conflict_warning,
            stacklevel=1,
        )
    else:
        os.environ["EPICS_CAS_INTF_ADDR_LIST"] = primary_ip
        os.environ["EPICS_CAS_BEACON_ADDR_LIST"] = primary_ip
        os.environ["EPICS_CAS_AUTO_BEACON_ADDR_LIST"] = "NO"


async def async_main() -> None:
    """Main entrypoint for virtac. Executed when running the 'virtac' command"""

    # Create an asyncio dispatcher
    loop = asyncio.get_running_loop()

    args = parse_arguments()
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, format=LOG_FORMAT)

    configure_ca()
    check_sim_params(args)

    # Determine the ring mode
    if args.ring_mode is not None:
        ring_mode = args.ring_mode
    else:
        try:
            ring_mode = str(os.environ["RINGMODE"])
        except KeyError:
            try:
                value = await caget("SR-CS-RING-01:MODE", timeout=1, format=2)
                ring_mode = cast(str, value.enums[int(value)])
                logging.warning(
                    "Ring mode not specified, using value stored in SR-CS-RING-01:MODE "
                    f"as the default: {ring_mode}"
                )
            except CANothing:
                ring_mode = "I04"
                logging.warning(f"Ring mode not specified, using default: {ring_mode}")

    # Create PVs.
    logging.debug("Creating ATIP server")
    server = await virtac_server.VirtacServer.create(
        ring_mode,
        DATADIR / ring_mode / "limits.csv",
        DATADIR / ring_mode / "bba.csv",
        DATADIR / ring_mode / "feedback.csv",
        DATADIR / ring_mode / "mirrored.csv",
        DATADIR / ring_mode / "tunefb.csv",
        args.linopt_function,
        args.disable_emittance,
        args.disable_chromaticity,
        args.disable_radiation,
        args.disable_tfb,
    )

    # Start the IOC.
    dispatcher = asyncio_dispatcher.AsyncioDispatcher(loop=loop)
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    # context = globals() | {"server": server}
    # softioc.interactive_ioc(context)

    while True:
        # logging.info("Sleeping")
        await asyncio.sleep(10)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
