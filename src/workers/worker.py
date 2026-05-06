import abc


class Worker(abc.ABC):
    @abc.abstractmethod
    def start(self):
        """Start the worker process."""
        raise NotImplementedError("Worker subclasses must implement the start method.")
