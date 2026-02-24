from abc import ABC, abstractmethod
from typing import Optional


class Handler(ABC):
    def __init__(self, next_handler: Optional["Handler"] = None):
        self.next_handler = next_handler

    def set_next(self, next_handler: "Handler") -> "Handler":
        """
        Sets next handler and returns it.

        :param next_handler: handler to be awoken next
        :type next_handler: Handler
        :return: Next handler
        :rtype: Handler
        """

        self.next_handler = next_handler
        return next_handler

    def handle(self, context: dict) -> dict:
        """Run this handler and pass the context to the next handler in the chain.

        :param context: current pipeline context shared between all handlers.
        :type context: dict
        :return: Updated context after this handler (and all subsequent handlers) have run.
        :rtype: dict
        """

        context = self.process(context)
        if self.next_handler:
            return self.next_handler.handle(context)
        return context

    @abstractmethod
    def process(self, context: dict) -> dict:
        """Perform handler-specific processing on the context.

        :param context: current pipeline context shared between all handlers.
        :type context: dict
        :return: Updated context after this handler's processing.
        :rtype: dict
        """

        ...
