from abc import ABC, abstractmethod

__author__ = 'frank.ma'


class Session(ABC):
    """
    your seat at one game, kept across rounds, as the play server sees it

    the server talks to a session only through view, act and advice, all plain JSON-ready dicts. A view is built
    from your seat's observation plus what has been made public (a hole card once it is turned over), so the
    browser is never sent a card your seat could not see: fair play holds by construction, not only by test
    """

    @abstractmethod
    def view(self) -> dict:
        """
        everything the table page draws: your cards, the public cards and facts, your legal actions, the result
        """
        pass

    @abstractmethod
    def act(self, action: dict) -> None:
        """
        apply one of your actions, then let the computer seats move until it is your turn or the round is over
        :raises ValueError: when the action is not legal now
        """
        pass

    @abstractmethod
    def advice(self) -> dict:
        """
        the coach's recommendation for your current decision and why, from your observation only
        """
        pass

    def progress(self) -> dict:
        """
        what to keep if the server restarts: plain numbers such as a bankroll or levels, never a card, so a saved file
        holds nothing hidden. A hand in play is not kept: bets settle when a hand ends, so it simply does not count
        """
        return {}

    def restore(self, progress: dict) -> None:
        """
        pick up from progress, between hands
        :raises (KeyError, TypeError, ValueError): when it is not progress this session could have saved
        """
        pass
