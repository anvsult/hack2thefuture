"""Animated 'claims racing to the cliff' chart (ECharts timeline spec).

Each open claim is a dot: x = days open, one lane per adjuster team, size = claim amount,
color = age zone (under warn / warn to critical / past critical). The browser plays one frame
per simulated day from the previous date to today, so dots slide toward the cliff.
"""
from __future__ import annotations

import zlib

import pandas as pd

from data import open_on

TEAMS = ["Team A", "Team B", "Team C"]
TEAM_CITIES = {"Team A": "Toronto, Ottawa", "Team B": "Calgary", "Team C": "Montreal, Quebec City"}
X_MAX = 60
GRAY, AMBER, RED = "#a8a7a1", "#eda100", "#e34948"
LANE_FILL = ["rgba(127,127,127,0.06)", "rgba(127,127,127,0.12)"]


def _jitter(claim_id: str) -> float:
    """Stable vertical offset inside the lane, so a dot never jumps between frames."""
    # -0.20..+0.38 of a lane: the top of each lane stays free for its title.
    return (zlib.crc32(claim_id.encode()) % 1000) / 1000 * 0.58 - 0.20


def _frame(claims: pd.DataFrame, day: pd.Timestamp, start: pd.Timestamp, crit: int,
           ticket_dates: dict[str, pd.Timestamp]) -> dict:
    op = claims[open_on(claims, day)]
    age = (day - op["reported_date"]).dt.days
    amount = op["claim_amount"].fillna(op["claim_amount"].median() if len(op) else 5000)
    data = []
    for cid, team, city, ctype, a, amt in zip(op["claim_id"], op["adjuster_team"], op["city"],
                                              op["claim_type"], age, amount):
        has_ticket = cid in ticket_dates and ticket_dates[cid] <= day
        item = {"id": cid, "name": cid,
                "symbolSize": round(8 + 22 * (min(float(amt), 30000) / 30000) ** 0.5, 1),
                "value": [min(int(a), X_MAX), TEAMS.index(team) + _jitter(cid), round(float(amt)),
                          cid, team, city, ctype, int(a), "Ticket sent" if has_ticket else "No ticket yet"]}
        if has_ticket:
            item["itemStyle"] = {"borderColor": "#ffffff", "borderWidth": 2}
        data.append(item)

    closed = claims[(claims["closed_date"] > start) & (claims["closed_date"] <= day)]
    closed_n = closed["adjuster_team"].value_counts()
    open_n = op["adjuster_team"].value_counts()
    lanes = []
    for i, team in enumerate(TEAMS):
        label = f"{team} ({TEAM_CITIES[team]})  ·  {open_n.get(team, 0)} open"
        if day > start:
            label += f"  ·  {closed_n.get(team, 0)} closed"
        lanes.append([{"yAxis": i - 0.48, "name": label, "itemStyle": {"color": LANE_FILL[i % 2]}},
                      {"yAxis": i + 0.48}])
    past = int((age >= crit).sum())
    return {
        "title": {"text": day.strftime("%a %b %-d, %Y"),
                  "subtext": f"{len(op)} open claims  ·  {past} past {crit} days"},
        "series": [{"data": data, "markArea": {"silent": True, "data": lanes,
                                                 "label": {"position": "insideTopLeft", "padding": [6, 8],
                                                           "color": "#7a7973", "fontSize": 12,
                                                           "fontWeight": "bold"}}}],
    }


def build_spec(claims: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp, warn: int, crit: int,
               ticket_dates: dict[str, pd.Timestamp], animate: bool = True, max_frames: int = 45) -> dict:
    """Frames from `start` (exclusive of nothing: start is frame 0) to `end`, one per day."""
    span = (end - start).days if start is not None else 0
    if animate and 0 < span <= max_frames:
        days = list(pd.date_range(start, end, freq="D"))
    else:
        days, start = [end], (start if start is not None and start < end else end)
    interval = 350 if len(days) <= 8 else 200          # a week plays in about 2.5 s, a month in about 6 s
    frames = [_frame(claims, d, start, crit, ticket_dates) for d in days]

    base = {
        "title": {"left": 8, "top": 0, "textStyle": {"fontSize": 22, "fontWeight": "bold"},
                  "subtextStyle": {"fontSize": 13}},
        "grid": {"left": 16, "right": 24, "top": 70, "bottom": 70 if len(days) > 1 else 40,
                 "containLabel": True},
        "xAxis": {"type": "value", "min": 0, "max": X_MAX, "interval": 10, "name": "Days open",
                  "nameLocation": "middle", "nameGap": 28,
                  "axisLabel": {"formatter": "{value}"}, "splitLine": {"show": False},
                  "axisLine": {"onZero": False}},
        "yAxis": {"type": "value", "min": -0.5, "max": 2.5, "show": False, "inverse": True},
        "tooltip": {"trigger": "item",
                    "formatter": "<b>{@[3]}</b><br/>{@[4]} · {@[5]} · {@[6]}<br/>"
                                 "Open {@[7]} days · ${@[2]}<br/>{@[8]}"},
        "visualMap": [
            {"type": "piecewise", "dimension": 0, "seriesIndex": 0, "orient": "horizontal",
             "right": 8, "top": 4, "itemGap": 14, "selectedMode": False,
             "pieces": [{"lt": warn, "color": GRAY, "label": f"Under {warn} days"},
                        {"gte": warn, "lt": crit, "color": AMBER, "label": f"{warn}-{crit} days (warn)"},
                        {"gte": crit, "color": RED, "label": f"{crit}+ days (critical)"}]},
        ],
        "series": [{
            "type": "scatter", "name": "Open claims",
            "animationDuration": 300, "animationEasing": "cubicOut",
            "animationDurationUpdate": min(interval - 30, 300), "animationEasingUpdate": "cubicOut",
            "emphasis": {"scale": 1.3},
            "markLine": {"silent": True, "symbol": "none", "animation": False, "data": [
                {"xAxis": warn, "lineStyle": {"color": AMBER, "type": "dashed", "width": 1},
                 "label": {"formatter": "warn", "color": AMBER, "position": "start", "distance": 6}},
                {"xAxis": crit, "lineStyle": {"color": RED, "type": "solid", "width": 2},
                 "label": {"formatter": "renewal cliff", "color": RED, "position": "start", "distance": 6,
                           "fontWeight": "bold"}},
            ]},
        }],
    }
    if len(days) == 1:
        # ECharts applies `options` only with a timeline; merge the single frame into the base.
        f = frames[0]
        base["title"] = {**base["title"], **f["title"]}
        base["series"][0] = {**base["series"][0], **f["series"][0]}
        return base
    if len(days) > 1:
        base["timeline"] = {"axisType": "category", "data": [d.strftime("%b %-d") for d in days],
                            "autoPlay": True, "loop": False, "playInterval": interval,
                            "bottom": 4, "left": 40, "right": 40,
                            "label": {"interval": "auto"}, "controlStyle": {"showPrevBtn": False,
                                                                            "showNextBtn": False}}
    return {"baseOption": base, "options": frames}
