"""Standard advantage games, sets, and win-by-two tiebreaks; independent of vision."""

from copy import deepcopy

from .models import Event, InitialScore


class TennisScore:
    def __init__(self, initial: InitialScore | None = None):
        self.initial = initial or InitialScore()
        self.points = list(self.initial.points)
        self.games = list(self.initial.games)
        self.sets = list(self.initial.sets)
        self.tiebreak = self.initial.tiebreak
        self.completed_sets = []
        self.winner = None
        self.total_points = [0, 0]

    def award(self, player: int):
        if player not in (0, 1):
            raise ValueError("Winner must be player 0 or 1.")
        if self.winner is not None:
            raise ValueError("The match has already finished.")
        self.total_points[player] += 1
        self.points[player] += 1
        other = 1 - player
        target = self.initial.tiebreak_target if self.tiebreak else 4
        if self.points[player] >= target and self.points[player] - self.points[other] >= 2:
            self.points = [0, 0]
            self.games[player] += 1
            if self.tiebreak or (
                self.games[player] >= 6 and self.games[player] - self.games[other] >= 2
            ):
                self.completed_sets.append(self.games.copy())
                self.sets[player] += 1
                self.games = [0, 0]
                self.tiebreak = False
                if self.sets[player] == (self.initial.best_of + 1) // 2:
                    self.winner = player
            elif self.games == [6, 6]:
                self.tiebreak = True
        elif not self.tiebreak and self.points == [4, 4]:
            self.points = [3, 3]
        return self.snapshot()

    def snapshot(self):
        if self.tiebreak:
            display = [str(n) for n in self.points]
        else:
            display = [["0", "15", "30", "40", "AD"][n] for n in self.points]
        return deepcopy(
            dict(
                points=display,
                games=self.games,
                sets=self.sets,
                total_points=self.total_points,
                tiebreak=self.tiebreak,
                completed_sets=self.completed_sets,
                winner=self.winner,
            )
        )


def timeline(initial: InitialScore, events: list[Event], confirmed_only=False):
    score = TennisScore(initial)
    rows = []
    for event in sorted(events, key=lambda e: (e.time, e.id)):
        if event.status == "rejected" or event.winner is None:
            continue
        if confirmed_only and event.status != "confirmed":
            continue
        if score.winner is not None:
            raise ValueError("Events continue after the match has finished; review the timeline.")
        rows.append(
            dict(
                time=event.time,
                event_id=event.id,
                status=event.status,
                score=score.award(event.winner),
            )
        )
    return rows, score.snapshot()
