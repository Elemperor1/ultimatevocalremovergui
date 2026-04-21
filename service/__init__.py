__all__ = ["DurableJobStore", "SeparationJobAPI", "SeparationJobRequest", "SeparationWorker"]


def __getattr__(name):
    if name in __all__:
        from service.api import DurableJobStore, SeparationJobAPI, SeparationJobRequest, SeparationWorker

        exported = {
            "DurableJobStore": DurableJobStore,
            "SeparationJobAPI": SeparationJobAPI,
            "SeparationJobRequest": SeparationJobRequest,
            "SeparationWorker": SeparationWorker,
        }
        globals()[name] = exported[name]
        return exported[name]
    raise AttributeError(f"module 'service' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | set(__all__))
