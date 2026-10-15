"""
revision_service.py  –  SM-2 Spaced Repetition Scheduler

SM-2 algorithm (SuperMemo 2):
  - quality   : 0–5 rating (derived from quiz score)
  - ease      : how easy the topic is (starts at 2.5)
  - interval  : days until next review
  - repetition: how many times reviewed successfully

Quality mapping from score %:
  0–19%  → 0 (complete blackout)
  20–39% → 1 (incorrect but familiar)
  40–59% → 2 (incorrect but easy to recall)
  60–69% → 3 (correct with difficulty)
  70–79% → 4 (correct after hesitation)
  80–100% → 5 (perfect recall)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import database as db


# ── SM-2 core ──────────────────────────────────────────────────────────────────
def _sm2(quality: int, repetitions: int,
         ease: float, interval: int) -> tuple[int, float, int]:
    """
    Apply one SM-2 step.
    Returns (new_interval_days, new_ease, new_repetitions).
    """
    if quality < 3:
        # Failed recall → reset
        new_reps     = 0
        new_interval = 1
    else:
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = round(interval * ease)
        new_reps = repetitions + 1

    # Update ease factor
    new_ease = ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease = max(1.3, new_ease)   # ease never drops below 1.3

    return new_interval, new_ease, new_reps


# ── Public API ─────────────────────────────────────────────────────────────────
def schedule_topic(session_id: str, topic: str, quality: int) -> None:
    """
    Update or create the SM-2 schedule for one topic after a quiz.
    quality: 0–5 (call pct_to_quality() to convert from score %).
    """
    # Load existing state from DB (if any)
    tasks = db.get_revision_tasks(session_id)
    existing = next(
        (t for t in tasks if t["topic"] == topic and t["status"] == "pending"),
        None
    )

    if existing:
        interval    = existing["interval_days"]
        ease        = existing["ease_factor"]
        repetitions = existing["repetitions"]
    else:
        interval    = 1
        ease        = 2.5
        repetitions = 0

    new_interval, new_ease, new_reps = _sm2(quality, repetitions, ease, interval)
    due_date = (date.today() + timedelta(days=new_interval)).isoformat()

    db.upsert_revision_task(
        session_id  = session_id,
        topic       = topic,
        due_date    = due_date,
        interval_days = new_interval,
        ease_factor   = round(new_ease, 3),
        repetitions   = new_reps,
    )


def pct_to_quality(pct: float) -> int:
    """Convert a quiz percentage to SM-2 quality (0–5)."""
    if pct >= 80:  return 5
    if pct >= 70:  return 4
    if pct >= 60:  return 3
    if pct >= 40:  return 2
    if pct >= 20:  return 1
    return 0


def get_due_today(session_id: str) -> List[Dict]:
    """Return pending tasks whose due_date is today or earlier."""
    today = date.today().isoformat()
    tasks = db.get_revision_tasks(session_id)
    return [t for t in tasks if t["status"] == "pending" and t["due_date"] <= today]


def get_schedule(session_id: str) -> Dict:
    """
    Return the full revision schedule split into sections:
        overdue, due_today, upcoming, completed
    """
    today  = date.today().isoformat()
    tasks  = db.get_revision_tasks(session_id)

    overdue   = [t for t in tasks if t["status"] == "pending"  and t["due_date"] <  today]
    due_today = [t for t in tasks if t["status"] == "pending"  and t["due_date"] == today]
    upcoming  = [t for t in tasks if t["status"] == "pending"  and t["due_date"] >  today]
    completed = [t for t in tasks if t["status"] == "done"]

    return {
        "overdue":   overdue,
        "due_today": due_today,
        "upcoming":  upcoming,
        "completed": completed,
        "total_pending": len(overdue) + len(due_today) + len(upcoming),
    }


def build_exam_plan(session_id: str, exam_date_str: str,
                    topic_history: List[Dict]) -> List[Dict]:
    """
    Given an exam date and topic history, create an initial revision plan
    that front-loads weak topics and spaces strong ones near the exam.
    Returns a list of scheduled task dicts (not yet saved — caller decides).
    """
    try:
        exam_date = date.fromisoformat(exam_date_str)
    except ValueError:
        exam_date = date.today() + timedelta(days=30)

    days_left = (exam_date - date.today()).days
    if days_left < 1:
        days_left = 1

    # Sort: weakest first
    sorted_topics = sorted(topic_history, key=lambda x: x["avg_pct"])

    plan   = []
    offset = 0
    step   = max(1, days_left // max(len(sorted_topics), 1))

    for t in sorted_topics:
        quality  = pct_to_quality(t["avg_pct"])
        interval, ease, reps = _sm2(quality, 0, 2.5, 1)
        # For weak topics: schedule earlier; strong: later
        if t["avg_pct"] < 60:
            due = (date.today() + timedelta(days=max(1, offset))).isoformat()
        else:
            due = (date.today() + timedelta(days=offset + step)).isoformat()

        plan.append({
            "topic":         t["topic"],
            "due_date":      due,
            "interval_days": interval,
            "ease_factor":   ease,
            "repetitions":   reps,
            "avg_pct":       t["avg_pct"],
        })
        offset += step

    return plan


def save_exam_plan(session_id: str, plan: List[Dict]) -> None:
    """Persist each task in the plan."""
    for item in plan:
        db.upsert_revision_task(
            session_id    = session_id,
            topic         = item["topic"],
            due_date      = item["due_date"],
            interval_days = item["interval_days"],
            ease_factor   = item["ease_factor"],
            repetitions   = item["repetitions"],
        )
