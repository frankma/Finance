import random
from typing import List

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.Play.BlackJackSession import BlackJackSession
from src.CardGames.Play.Lesson import Lesson, Step

__author__ = 'frank.ma'

LOW = (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)


def _stack(session: BlackJackSession, order: str) -> None:
    # deal these cards next, in order: you, the dealer's up card, you, the dealer's hole card, then any draws
    cards = Card.parse_many(order)
    rest = list(session.table.shoe.cards)
    for card in cards:
        rest.remove(card)
    session.table.shoe.cards = rest + list(reversed(cards))


def _dealt(order: str, bet: int = 10):
    # a table with this hand already dealt, waiting for your first decision
    def setup(rng: random.Random) -> BlackJackSession:
        session = BlackJackSession({'rules': 's17'}, rng)
        _stack(session, order)
        session.act({'type': 'deal', 'bet': bet})
        return session
    return setup


def _low_cards_shown(n: int):
    # a table where the rounds since the shuffle have shown n low cards, so the count is high, waiting for a bet
    def setup(rng: random.Random) -> BlackJackSession:
        session = BlackJackSession({'rules': 's17'}, rng)
        shoe = session.table.shoe
        low = [card for card in shoe.cards if card.rank in LOW][:n]
        rest = list(shoe.cards)
        for card in low:
            rest.remove(card)
        shoe.cards = rest + low
        shoe.draw_many(n)  # face up, so the count sees them
        session.log.append('Earlier rounds since the shuffle showed %i low cards (2 to 6) and few high ones.' % n)
        return session
    return setup


def _move(name: str) -> dict:
    return {'type': 'move', 'move': name}


def lessons() -> List[Lesson]:
    return [
        Lesson('bust', 'Let the dealer bust', 'Stand on a weak total when the dealer shows a weak card.',
               _dealt('9H 6S 4C TD KC'), (
                   Step(_move('stand'),
                        'You have hard 13 and the dealer shows a 6. The dealer must hit until 17, and a 6 is the '
                        'worst card to start from. Hit or stand?',
                        'Right: stand. With a 6 showing the dealer busts about 4 times in 10, and hitting 13 would '
                        'bust you with any 9 or higher. Let the dealer take the risk.',
                        'Hitting 13 busts you with any 9, 10, J, Q or K, about 3 cards in 10, while the dealer, stuck '
                        'drawing from a 6, busts about 4 times in 10. Try standing.'),
               ), 'Basic strategy stands on 13 to 16 against a dealer 2 to 6, except 12 against a 2 or 3.',
               ('dealer', 'basic-strategy')),
        Lesson('sixteen', 'Sixteen against a ten', 'The hardest hand in blackjack, and why you hit it.',
               _dealt('TS TH 6D 7C 4S'), (
                   Step(_move('hit'),
                        'You have hard 16 and the dealer shows a 10. Standing only wins if the dealer busts. Hit or '
                        'stand?',
                        'Right: hit. With a 10 showing the dealer busts only about 1 time in 4, so standing on 16 '
                        'usually loses. Hitting loses too, but a little less often.',
                        'Standing on 16 wins only when the dealer busts, about 1 time in 4 with a 10 showing. Hitting '
                        'gives you a chance to make a real hand. Try hitting.'),
                   Step(_move('stand'),
                        'You drew a 4: now you have 20. Hit or stand?',
                        'Right: stand. 20 beats everything but 21, and any card but an ace would bust you.',
                        'Only an ace keeps 20 from busting. Stand on 20.'),
               ), '16 against a 10 loses either way over time; the skill is losing as little as possible.',
               ('moves', 'basic-strategy')),
        Lesson('double', 'Double down on 11', 'Put more money in when the next card is likely to help.',
               _dealt('6D 9H 5C 8S TC'), (
                   Step(_move('double'),
                        'You have 11 and the dealer shows a 9. Any ten-value card gives you 21. Hit, stand or double?',
                        'Right: double. About 4 cards in 13 are worth ten, and almost any card leaves you with a '
                        'strong total, so you are the favourite and should have twice the bet riding on it.',
                        'With 11, one more card very often makes a strong hand, and doubling takes exactly one card '
                        'for twice the bet. When you are the favourite, bet more. Try doubling.'),
               ), 'Double 11 against anything but an ace, and 10 against 2 to 9.', ('moves', 'basic-strategy')),
        Lesson('soft', 'Soft 18 against a 9', 'An ace counting 11 lets you hit without risk of busting.',
               _dealt('AH 9C 7S 9D 2C'), (
                   Step(_move('hit'),
                        'You have an ace and a 7: soft 18, since the ace counts 11. The dealer shows a 9. 18 feels '
                        'safe, but is it enough? Hit or stand?',
                        'Right: hit. Against a 9, 18 loses more often than it wins, and one card cannot bust a soft '
                        'hand: if the ace has to drop to 1, you still have a hand to play.',
                        'A soft hand cannot bust on the next card, because the ace drops from 11 to 1 if it must. '
                        'Against a 9, 18 is not good enough to stand on. Try hitting.'),
                   Step(_move('stand'),
                        'You drew a 2: soft 20. Hit or stand?',
                        'Right: stand on 20.',
                        '20 is a strong hand: stand.'),
               ), 'Soft hands give you a free look at another card; use it when your total would lose.',
               ('values', 'basic-strategy')),
        Lesson('split', 'Always split 8s', 'Turn one bad hand into two decent ones.',
               _dealt('8S TH 8D 7D TC 9C'), (
                   Step(_move('split'),
                        'You have two 8s, 16 in all, and the dealer shows a 10. Hit, stand or split?',
                        'Right: split. 16 is the worst total in the game, but 8 is a fine card to start a hand from. '
                        'Splitting makes two hands, each with its own bet, starting from 8.',
                        '16 is the worst total you can hold. Two hands starting from 8 each do much better. Try '
                        'splitting.'),
                   Step(_move('stand'),
                        'Your first hand drew a 10: 18 against the dealer\'s 10. Hit or stand?',
                        'Right: stand on 18.',
                        '18 busts on any card over 3. Stand.'),
                   Step(_move('stand'),
                        'Your second hand drew a 9: 17. Hit or stand?',
                        'Right: stand on 17. Hard 17 or more always stands.',
                        'Hard 17 busts with any card over 4. Basic strategy always stands on hard 17. Stand.'),
               ), 'Always split aces and 8s; never split 10s or 5s.', ('splitting', 'basic-strategy')),
        Lesson('count', 'Bet more when the count is high', 'Use the cards already shown to size your bet.',
               _low_cards_shown(26), (
                   Step({'type': 'deal', 'bet_at_least': 10},
                        'Since the shuffle, many low cards have been shown and few high ones: look at the hi-lo '
                        'count in the panel. The rest of the shoe is rich in tens and aces, which favour you. How much '
                        'will you bet? Ask the coach (C) for the Kelly bet if you like.',
                        'Right: the count is high, so the edge is yours and the bet goes up. The Kelly rule stakes '
                        'edge divided by variance of your bankroll; ask the coach to see the number it uses.',
                        'With a true count this high the edge is with you, not the house, so a minimum bet wastes '
                        'it. Bet at least twice the minimum, or ask the coach (C) for its Kelly bet.'),
               ), 'Play the hand out as you like. When the count is low, go back to the minimum.',
               ('counting', 'kelly')),
    ]
