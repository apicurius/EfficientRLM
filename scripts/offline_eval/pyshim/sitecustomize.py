"""Compat shim for the canary harness (ai16). Loaded via PYTHONPATH.

The prime-rl vllm plugin (prime_rl.inference.patches.transformers_v5_compat)
replaces LoRAModel.from_local_checkpoint with a body copied from an older
vLLM, whose signature predates the `moe_ep_spec` kwarg the installed vLLM's
callers now pass. prime-rl source is off-limits by standing rule, so this
shim wraps the plugin entry point post-import and re-wraps the patched
classmethod to swallow kwargs it does not know. Safe here because our
adapters are attention-only LoRA (no MoE expert weights).
Inert in any process that never imports prime_rl.inference.patches.
"""
import sys
from importlib import abc, util

_TARGET = "prime_rl.inference.patches"


def _make_tolerant():
    import functools
    from vllm.lora.lora_model import LoRAModel

    inner = LoRAModel.from_local_checkpoint.__func__

    @functools.wraps(inner)
    def tolerant(cls, *args, **kwargs):
        kwargs.pop("moe_ep_spec", None)
        return inner(cls, *args, **kwargs)

    LoRAModel.from_local_checkpoint = classmethod(tolerant)


def _wrap_module(module):
    orig = module.transformers_v5_compat

    def transformers_v5_compat():
        orig()
        _make_tolerant()

    module.transformers_v5_compat = transformers_v5_compat


class _LoaderWrap(abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _wrap_module(module)


class _Finder(abc.MetaPathFinder):
    _busy = False

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET or _Finder._busy:
            return None
        _Finder._busy = True
        try:
            spec = util.find_spec(fullname)
        finally:
            _Finder._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _LoaderWrap(spec.loader)
        return spec


sys.meta_path.insert(0, _Finder())
