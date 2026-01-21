from abc import ABC, abstractmethod


class Handler(ABC):
    def __init__(self, next_handler=None):
        self.next_handler = next_handler

    def handle(self, context: dict) -> dict:
        context = self.process(context)
        if self.next_handler:
            return self.next_handler.handle(context)
        return context

    @abstractmethod
    def process(self, context: dict) -> dict:
        pass
