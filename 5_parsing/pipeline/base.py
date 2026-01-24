from abc import ABC, abstractmethod
from typing import Optional


class Handler(ABC):
    def __init__(self, next_handler: Optional["Handler"] = None):
        self.next_handler = next_handler

    def set_next(self, next_handler: "Handler") -> "Handler":
        """
        Устанавливает следующий обработчик и возвращает его,
        чтобы можно было писать:
        loader.set_next(cleaner).set_next(basic) ...
        """
        self.next_handler = next_handler
        return next_handler

    def handle(self, context: dict) -> dict:
        context = self.process(context)
        if self.next_handler:
            return self.next_handler.handle(context)
        return context

    @abstractmethod
    def process(self, context: dict) -> dict:
        ...
