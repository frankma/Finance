import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'


@dataclass(frozen=True)
class Step(object):
    """
    one decision in a lesson: the action it waits for, what to look at before choosing, why the move is right, and
    a hint when you choose otherwise

    expect matches an action by its fields, e.g. {'type': 'move', 'move': 'hit'}; the key 'bet_at_least' asks for a
    deal with at least that bet
    """
    expect: dict
    prompt: str
    why: str
    hint: str

    def matches(self, action: dict) -> bool:
        for key, value in self.expect.items():
            if key == 'bet_at_least':
                try:
                    if int(action.get('bet', 0)) < value:
                        return False
                except (TypeError, ValueError):
                    return False
            elif action.get(key) != value:
                return False
        return True


@dataclass(frozen=True)
class Lesson(object):
    """
    a table set up for one idea, usually by stacking the cards, and the steps that walk you through it

    the stacked cards only decide what is dealt: the table still shows you only what your seat could see, so the
    prompts never mention a face-down card before it is turned over
    """
    key: str
    title: str
    summary: str
    setup: Callable[[random.Random], Session]
    steps: Tuple[Step, ...]
    wrap: str = ''  # a closing remark once every step is done
    tags: Tuple[str, ...] = field(default=())  # sections of the game's rules this lesson practises


class LessonSession(Session):
    """
    a game's lessons, one table at a time: each of your actions is checked against the step in front of you; the
    right one is played and explained, any other is not played and gets a hint, so a lesson always goes as written.
    Once its steps are done, the hand can be played out freely
    """

    def __init__(self, lessons: List[Lesson], rng: Optional[random.Random] = None):
        if not lessons:
            raise ValueError('need at least one lesson.')
        self.lessons = lessons
        self.rng = rng if rng is not None else random.Random()
        self.done = [False] * len(lessons)
        self.index = 0
        self.step = 0
        self.mistakes = 0
        self.feedback = None  # type: Optional[dict]
        self.table = None  # type: Optional[Session]
        self._start(0)

    @property
    def lesson(self) -> Lesson:
        return self.lessons[self.index]

    @property
    def complete(self) -> bool:
        return self.step >= len(self.lesson.steps)

    def _start(self, index: int) -> None:
        if not 0 <= index < len(self.lessons):
            raise ValueError('lesson %r is not one of the %i lessons.' % (index, len(self.lessons)))
        self.index, self.step, self.mistakes, self.feedback = index, 0, 0, None
        self.table = self.lessons[index].setup(random.Random(self.rng.random()))

    def act(self, action: dict) -> None:
        kind = action.get('type')
        if kind == 'lesson':
            try:
                self._start(int(action.get('index')))
            except (TypeError, ValueError):
                raise ValueError('pick a lesson by its number, from 0 to %i.' % (len(self.lessons) - 1))
            return
        if kind == 'next':
            self._start(min(self.index + 1, len(self.lessons) - 1))
            return
        if self.complete:
            self.table.act(action)  # the lesson is done: play the hand out as you like
            return
        step = self.lesson.steps[self.step]
        if not step.matches(action):
            self.mistakes += 1
            self.feedback = {'kind': 'hint', 'text': step.hint}
            return
        self.table.act(action)
        self.step += 1
        text = step.why
        if self.complete:
            self.done[self.index] = True
            text = (text + ' ' + self.lesson.wrap).strip()
        self.feedback = {'kind': 'right', 'text': text}

    def view(self) -> dict:
        out = self.table.view()
        lesson = self.lesson
        step = None if self.complete else lesson.steps[self.step]
        out['lesson'] = {
            'index': self.index, 'title': lesson.title, 'summary': lesson.summary, 'tags': list(lesson.tags),
            'step': self.step, 'n_steps': len(lesson.steps), 'prompt': step.prompt if step else None,
            'awaiting': step.expect.get('type') if step else None, 'feedback': self.feedback,
            'complete': self.complete, 'mistakes': self.mistakes,
            'lessons': [{'title': x.title, 'done': d} for x, d in zip(self.lessons, self.done)],
        }
        return out

    def advice(self) -> dict:
        return self.table.advice()
