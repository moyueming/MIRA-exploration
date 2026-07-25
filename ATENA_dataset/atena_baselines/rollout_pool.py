import multiprocessing as mp


class PersistentRolloutExecutor:
    def __init__(
        self,
        workers,
        worker_function,
        context_factory=None,
        maxtasksperchild=64,
    ):
        self.workers = max(1, int(workers))
        self.worker_function = worker_function
        self.context_factory = context_factory or mp.get_context
        self.maxtasksperchild = max(2, int(maxtasksperchild))
        self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(terminate=exc_type is not None)
        return False

    def map(self, worker_args):
        worker_args = list(worker_args)
        if self.workers <= 1 or len(worker_args) <= 1:
            return [self.worker_function(arg) for arg in worker_args]
        if self.pool is None:
            context = self.context_factory("spawn")
            self.pool = context.Pool(
                processes=min(self.workers, len(worker_args)),
                maxtasksperchild=self.maxtasksperchild,
            )
        return self.pool.map(self.worker_function, worker_args)

    def close(self, terminate=False):
        if self.pool is None:
            return
        pool = self.pool
        self.pool = None
        if terminate:
            pool.terminate()
        else:
            pool.close()
        pool.join()
