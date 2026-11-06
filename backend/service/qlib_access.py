from functools import wraps


def run_with_qlib_read_access(callback, *args, **kwargs):
    from core.adapters.qlib.generation import qlib_generation_read_lock

    with qlib_generation_read_lock():
        return callback(*args, **kwargs)


def qlib_read_locked(callback):
    @wraps(callback)
    def wrapped(*args, **kwargs):
        return run_with_qlib_read_access(callback, *args, **kwargs)

    return wrapped


def qlib_mutation_locked(callback):
    @wraps(callback)
    def wrapped(*args, **kwargs):
        from core.adapters.qlib.generation import qlib_generation_mutation_lock

        with qlib_generation_mutation_lock():
            return callback(*args, **kwargs)

    return wrapped
