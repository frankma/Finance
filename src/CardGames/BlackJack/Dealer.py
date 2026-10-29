from src.CardGames.BlackJack.Hand import Hand
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.Core.Shoe import Shoe

__author__ = 'frank.ma'


class Dealer(object):
    def __init__(self, rules: Rules):
        self.rules = rules

    def should_hit(self, hand: Hand) -> bool:
        if hand.total < 17:
            return True
        return hand.total == 17 and hand.is_soft and self.rules.dealer_hits_soft_17

    def play(self, hand: Hand, shoe: Shoe) -> Hand:
        while self.should_hit(hand):
            hand.add(shoe.draw())
        return hand
