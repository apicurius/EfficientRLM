"""Switch the Lightning studio to a CPU machine to stop GPU billing.

GUARDED: switching machines reboots the studio and kills any live process,
so this refuses to act while the control run's wandb state is "running".
Run it after the step-200 note (or if the run is confirmed dead):

    uv run --with lightning_sdk --with wandb python scripts/studio_to_cpu.py --list
    STUDIO_NAME=<name> uv run --with lightning_sdk --with wandb python scripts/studio_to_cpu.py
    STUDIO_NAME=<name> FORCE=1 ... python scripts/studio_to_cpu.py   # skip the wandb guard

Credentials: ~/.lightning/credentials.json (already present on kuvalar);
wandb key read from /scratch/omeerdogan23/erlm/rlm/.env.
"""
import os
import re
import sys

WANDB_RUN = "omeerdogan-koc-university/rlm-qwen3-30b/4f864caba69d400eaa4d45bee418ba1f"
ENV_FILE = "/scratch/omeerdogan23/erlm/rlm/.env"


def wandb_state() -> str:
    import wandb

    key = re.search(r"^WANDB_API_KEY=(.+)$", open(ENV_FILE).read(), re.M).group(1)
    return wandb.Api(api_key=key).run(WANDB_RUN).state


def main() -> None:
    from lightning_sdk import Machine, Studio

    if "--list" in sys.argv:
        from lightning_sdk import Teamspace

        ts = Teamspace()
        for s in ts.studios:
            print(f"{s.name}: {s.status} machine={getattr(s, 'machine', '?')}")
        return

    name = os.environ.get("STUDIO_NAME")
    if not name:
        sys.exit("set STUDIO_NAME (see --list)")

    if os.environ.get("FORCE") != "1":
        state = wandb_state()
        if state == "running":
            sys.exit(f"REFUSING: control run wandb state is '{state}' — switching would kill it. FORCE=1 to override.")
        print(f"wandb guard passed (run state: {state})")

    studio = Studio(name=name)
    print(f"studio {name}: {studio.status}")
    studio.switch_machine(Machine.CPU)  # blocks until CPU machine is up
    print(f"switched to CPU: {studio.status}")


if __name__ == "__main__":
    main()
