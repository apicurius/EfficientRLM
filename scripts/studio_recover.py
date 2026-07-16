"""Recover the studio after a reboot/GPU loss: start it and invoke ~/recover.sh.

Counterpart of the recovery contract in mailbox/inbox/006-recovery-contract.md:
the studio agent maintains an idempotent ~/recover.sh (tmux resume of training
+ restart of the studio Claude session pointed at the HF mailbox). This script
only starts the machine and calls that entry point.

    STUDIO_NAME=<name> uv run --with lightning_sdk python scripts/studio_recover.py
    # optional: MACHINE=RTX_PRO_6000 to pin the GPU type on restart

Safe by construction: Studio.start() is a no-op if already running, and
recover.sh is required to be idempotent (checks tmux sessions before spawning).
"""
import os
import sys


def main() -> None:
    from lightning_sdk import Machine, Studio

    name = os.environ.get("STUDIO_NAME")
    if not name:
        sys.exit("set STUDIO_NAME")

    studio = Studio(name=name)
    print(f"studio {name}: {studio.status}")

    if str(studio.status) != "Status.Running":
        machine = os.environ.get("MACHINE")
        if machine:
            studio.start(machine=getattr(Machine, machine))
        else:
            studio.start()  # last-used machine type
        print(f"started: {studio.status}")

    out = studio.run("bash ~/recover.sh 2>&1 | tail -20")
    print(out)


if __name__ == "__main__":
    main()
