"""Traceback-free quarantine state; never retain caller locals or secret text."""


def retained_failure() -> RuntimeError:
    # Deliberately accept no exception: neither its message nor its traceback,
    # cause, context or notes belong in a process-lifetime ownership registry.
    return RuntimeError("Resource ownership is quarantined until process restart after an uncertain cleanup.")
