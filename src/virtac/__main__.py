import asyncio
import logging
import os
import socket
import threading
from argparse import ArgumentError, ArgumentParser
from concurrent.futures import Future
from pathlib import Path
from typing import cast
from warnings import warn

from aioca import CANothing, caget
from atip.simulator import SimParams
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


async def async_main(
    server_ready: Future[virtac_server.VirtacServer],
    stop_requested: threading.Event,
) -> None:
    """Main entrypoint for virtac. Executed when running the 'virtac' command"""

    # Create the asyncio dispatcher for the IOC using the running loop
    loop = asyncio.get_running_loop()
    dispatcher = asyncio_dispatcher.AsyncioDispatcher(loop)

    args = parse_arguments()
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, format=LOG_FORMAT)

    configure_ca()

    sim_params = SimParams(
        args.linopt_function,
        not args.disable_emittance,
        not args.disable_chromaticity,
        not args.disable_radiation,
    )

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

    # Create Virtac server
    logging.debug("Creating ATIP server")
    server = await virtac_server.VirtacServer.create(
        ring_mode,
        DATADIR / ring_mode / "limits.csv",
        DATADIR / ring_mode / "bba.csv",
        DATADIR / ring_mode / "feedback.csv",
        DATADIR / ring_mode / "mirrored.csv",
        DATADIR / ring_mode / "tunefb.csv",
        sim_params,
        args.disable_tfb,
    )

    # Start the IOC.
    builder.LoadDatabase()
    softioc.iocInit(dispatcher, enable_pva=False)

    # Allow the main thread to continue and return server
    server_ready.set_result(server)

    # Keep this asyncio loop alive otherwise the IOC will die
    while not stop_requested.is_set():
        await asyncio.sleep(1)


def main() -> None:
    server_ready: Future[virtac_server.VirtacServer] = Future()
    stop_requested = threading.Event()

    def run_async_app() -> None:
        asyncio.run(async_main(server_ready, stop_requested))

    # We start the IOC in its own thread, which allows the main thread to be
    # used for the interactive shell
    worker = threading.Thread(target=run_async_app, name="virtac-asyncio")
    worker.start()

    # Wait for the server to be initialised so we can pass it to the interactive
    # shell
    server = server_ready.result()

    context = globals() | {"server": server}
    softioc.interactive_ioc(context, call_exit=False)

    # Cleanup after exit
    stop_requested.set()
    worker.join()


if __name__ == "__main__":
    main()
