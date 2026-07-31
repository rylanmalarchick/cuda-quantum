# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define a CUDA-Q runtime endpoint in Python.

A runtime endpoint is the *launch* half of a backend: it receives an already
compiled kernel and executes it. It says nothing about how the kernel was
compiled -- that remains the job of the active target, selected as usual with
``cudaq.set_target``.

An endpoint is any plain Python object implementing one or more of the
protocols below. Register it with :func:`set_runtime_endpoint`::

    import cudaq
    from cudaq._experimental import SampleResult, set_runtime_endpoint

    class MyEndpoint:

        def sample(self, module, arguments, *, shots_count, **options):
            submit_somewhere(module, list(arguments))
            return SampleResult({"00": shots_count})

    endpoint = MyEndpoint()
    set_runtime_endpoint(endpoint)
    cudaq.sample(my_kernel)      # dispatched to endpoint.sample

Calling ``cudaq.set_target(...)`` or ``cudaq.reset_target()`` replaces the
platform's QPUs and thereby removes the endpoint again; there is no separate
uninstall call.

.. warning::

   This API is experimental. There is currently no way for the runtime to check
   that the active target's compilation settings produce IR the endpoint
   understands, so that contract is entirely yours to uphold. Registering an
   endpoint also makes the platform fall back to a default local-simulator
   compile target, which it reports with a warning.
"""

from typing import Protocol, runtime_checkable

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import (
    CompiledModule,
    KernelArgs,
    ObserveResult,
    SampleResult,
    SpinOperator,
    set_runtime_endpoint as _set_runtime_endpoint,
)

__all__ = [
    "CompiledModule",
    "KernelArgs",
    "ObserveResult",
    "SampleResult",
    "SpinOperator",
    "SupportsObserve",
    "SupportsRun",
    "SupportsSample",
    "set_runtime_endpoint",
]

# Every launch method receives the compiled kernel and its arguments, plus
# policy-derived keyword arguments. Accept `**kwargs` so that keyword arguments
# added by later CUDA-Q releases do not break existing endpoints.
#
# `module` is yours to keep; `args` is a view onto memory owned by the caller
# and must not be used after the method returns.


@runtime_checkable
class SupportsSample(Protocol):
    """An endpoint that can serve ``cudaq.sample``."""

    def sample(self, module: CompiledModule, args: KernelArgs,
               **kwargs) -> SampleResult:
        """Execute a compiled kernel and return its measurement counts.

        Keyword arguments: ``shots_count`` (int) and ``explicit_measurements``
        (bool).
        """
        ...


@runtime_checkable
class SupportsObserve(Protocol):
    """An endpoint that can serve ``cudaq.observe``."""

    def observe(self, module: CompiledModule, args: KernelArgs,
                **kwargs) -> ObserveResult:
        """Execute a compiled kernel and return an expectation value.

        Keyword arguments: ``spin_operator`` (:class:`SpinOperator`) and, when
        the launch specifies one, ``shots_count`` (int).
        """
        ...


@runtime_checkable
class SupportsRun(Protocol):
    """An endpoint that can serve ``cudaq.run``.

    Note that a local endpoint is invoked once per shot, and that the return
    value is a raw QIR output log which CUDA-Q decodes into the kernel's return
    values.
    """

    def run(self, module: CompiledModule, args: KernelArgs, **kwargs) -> str:
        """Execute a compiled kernel and return its QIR output log.

        Keyword arguments: ``shots_count`` (int).
        """
        ...


_PROTOCOLS = (SupportsSample, SupportsObserve, SupportsRun)


def set_runtime_endpoint(endpoint, qpu_id: int = 0) -> None:
    """Route kernel launches to `endpoint` instead of the active target's QPU.

    Args:
      endpoint: An object implementing at least one of :class:`SupportsSample`,
        :class:`SupportsObserve` or :class:`SupportsRun`. Launches under a
        policy the object does not implement raise a `RuntimeError`.
      `qpu_id` (Optional[int]): The QPU on the current platform whose launches
        to redirect. Defaults to zero.

    Raises:
      TypeError: If `endpoint` implements none of the protocols.
    """
    if not any(isinstance(endpoint, protocol) for protocol in _PROTOCOLS):
        raise TypeError(
            f"{type(endpoint).__name__} is not a runtime endpoint: it must "
            f"define at least one of "
            f"{', '.join(p.__name__ for p in _PROTOCOLS)}.")
    _set_runtime_endpoint(endpoint, qpu_id)
