"""
agent_service.py  –  Agentic Study Coach
Rule-based agent that reads real quiz history and produces a study plan.
No LLM required. Every recommendation is traceable to actual data.

Agent loop (7 stages):
  1. Observe   – load attempt history and topic scores from DB
  2. Analyse   – classify topics as weak / improving / strong
  3. Plan      – build ordered revision list
  4. Select    – choose the next best action
  5. Execute   – return action payload (quiz params, notes excerpt, message)
  6. Evaluate  – compare latest attempt against previous
  7. Update    – revise priorities in the revision schedule
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import database as db
from revision_service import schedule_topic, get_due_today


# ── Constants ──────────────────────────────────────────────────────────────────
WEAK_THRESHOLD       = 60   # % below this → weak
STRONG_THRESHOLD     = 80   # % above this → strong
MIN_ATTEMPTS_TREND   = 2    # need ≥ N attempts before calling "improving"


# ── Stage 1 — Observe ─────────────────────────────────────────────────────────
def observe(session_id: str) -> Dict:
    """Load the student's complete learning history from the database."""
    attempts     = db.get_attempts(session_id, limit=20)
    topic_hist   = db.get_topic_history(session_id)
    goal         = db.get_goal(session_id)
    due_today    = get_due_today(session_id)
    return {
        "attempts":   attempts,
        "topic_hist": topic_hist,
        "goal":       goal,
        "due_today":  due_today,
    }


# ── Stage 2 — Analyse ─────────────────────────────────────────────────────────
def analyse(observation: Dict) -> Dict:
    """
    Classify each topic and produce insight messages grounded in real data.
    Returns:
        weak      – topics with avg_pct < WEAK_THRESHOLD
        improving – weak topics showing upward trend
        strong    – topics with avg_pct >= STRONG_THRESHOLD
        insights  – human-readable bullet points
    """
    topic_hist = observation["topic_hist"]
    attempts   = observation["attempts"]

    weak, improving, strong = [], [], []
    insights: List[str] = []

    for t in topic_hist:
        avg  = t["avg_pct"]
        last = t["last_pct"]
        n    = t["attempts"]
        name = t["topic"]

        if avg >= STRONG_THRESHOLD:
            strong.append(t)
            insights.append(
                f"✅ <strong>{name}</strong>: solid at {avg}% average — keep it warm."
            )
        elif t["trend"] == "improving" and n >= MIN_ATTEMPTS_TREND:
            improving.append(t)
            insights.append(
                f"📈 <strong>{name}</strong>: improving ({avg}% avg → {last}% last) "
                f"— keep the momentum."
            )
        else:
            weak.append(t)
            reason = _weak_reason(avg, n)
            insights.append(
                f"⚠️ <strong>{name}</strong>: {avg}% average across {n} attempt(s). "
                f"{reason}"
            )

    # Overall streak
    if len(attempts) >= 2:
        delta = attempts[0]["score_pct"] - attempts[1]["score_pct"]
        if delta >= 10:
            insights.insert(0, f"🔥 Overall score improved by {delta:.0f}% since last quiz!")
        elif delta <= -10:
            insights.insert(0, f"📉 Overall score dropped by {abs(delta):.0f}% — let's focus.")

    # No history yet
    if not topic_hist:
        insights = ["📚 No quiz history yet. Take your first quiz to get personalised recommendations!"]

    return {
        "weak":      weak,
        "improving": improving,
        "strong":    strong,
        "insights":  insights,
    }


def _weak_reason(avg_pct: float, attempts: int) -> str:
    if attempts == 1:
        return "First attempt — more practice needed."
    if avg_pct < 40:
        return "Consistently low — prioritise before exam."
    if avg_pct < 60:
        return "Below passing threshold — schedule a revision session."
    return "Just below target — a focused review should help."


# ── Stage 3 — Plan ────────────────────────────────────────────────────────────
def build_plan(analysis: Dict, observation: Dict) -> List[Dict]:
    """
    Return an ordered revision list.
    Priority: weak (lowest avg first) > improving > strong.
    Includes the SM-2 due date for each topic.
    """
    tasks = db.get_revision_tasks(observation.get("session_id", ""))

    due_map: Dict[str, str] = {}
    for t in tasks:
        due_map[t["topic"]] = t["due_date"]

    plan: List[Dict] = []

    for t in sorted(analysis["weak"], key=lambda x: x["avg_pct"]):
        plan.append({
            "topic":    t["topic"],
            "priority": "high",
            "reason":   f"Weak area — {t['avg_pct']}% average",
            "due":      due_map.get(t["topic"], "Today"),
        })

    for t in analysis["improving"]:
        plan.append({
            "topic":    t["topic"],
            "priority": "medium",
            "reason":   f"Improving — maintain momentum",
            "due":      due_map.get(t["topic"], _days_from_now(3)),
        })

    for t in analysis["strong"]:
        plan.append({
            "topic":    t["topic"],
            "priority": "low",
            "reason":   f"Strong — periodic review only",
            "due":      due_map.get(t["topic"], _days_from_now(7)),
        })

    return plan


def _days_from_now(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


# ── Stage 4 — Select action ───────────────────────────────────────────────────
ACTION_QUIZ          = "quiz"
ACTION_REVIEW_NOTES  = "review_notes"
ACTION_REST          = "rest"
ACTION_CELEBRATE     = "celebrate"

def select_action(analysis: Dict, observation: Dict) -> Dict:
    """
    Choose the single best next action based on analysis.
    Returns: {action, topic, reason, quiz_params}
    """
    due_today = observation.get("due_today", [])

    # Overdue / due today — highest priority
    if due_today:
        topic = due_today[0]["topic"]
        return {
            "action":  ACTION_QUIZ,
            "topic":   topic,
            "reason":  f"Revision due today for <strong>{topic}</strong>.",
            "quiz_params": {"topic_filter": topic, "num_questions": 5},
        }

    # Weakest topic
    if analysis["weak"]:
        weakest = analysis["weak"][0]
        return {
            "action":  ACTION_QUIZ,
            "topic":   weakest["topic"],
            "reason":  f"Focus on <strong>{weakest['topic']}</strong> "
                       f"({weakest['avg_pct']}% avg) — your lowest-scoring area.",
            "quiz_params": {"topic_filter": weakest["topic"], "num_questions": 5},
        }

    # All improving — keep going
    if analysis["improving"] and not analysis["weak"]:
        topic = analysis["improving"][0]["topic"]
        return {
            "action":  ACTION_QUIZ,
            "topic":   topic,
            "reason":  f"<strong>{topic}</strong> is improving — a focused quiz will consolidate it.",
            "quiz_params": {"topic_filter": topic, "num_questions": 5},
        }

    # Everything strong
    if analysis["strong"] and not analysis["weak"] and not analysis["improving"]:
        return {
            "action":  ACTION_CELEBRATE,
            "topic":   None,
            "reason":  "All topics are strong. Consider attempting a full-length practice exam.",
            "quiz_params": {"num_questions": 15},
        }

    # No history
    return {
        "action":  ACTION_REVIEW_NOTES,
        "topic":   None,
        "reason":  "Complete your first quiz to receive personalised recommendations.",
        "quiz_params": {"num_questions": 10},
    }


# ── Stage 5 — Execute ─────────────────────────────────────────────────────────
def execute_action(action: Dict, topics_text: Dict[str, str]) -> Dict:
    """
    Prepare the payload the frontend needs to run the action.
    For QUIZ: returns the topic text slice to generate questions from.
    For REVIEW_NOTES: returns the relevant passage.
    """
    topic  = action.get("topic")
    result = {"action": action["action"], "topic": topic, "reason": action["reason"]}

    if action["action"] in (ACTION_QUIZ, ACTION_REVIEW_NOTES):
        if topic and topic in topics_text:
            passage = topics_text[topic][:800]
            result["passage"] = passage
        else:
            # Fallback to full text excerpt
            all_text = " ".join(topics_text.values())
            result["passage"] = all_text[:800]

    result["quiz_params"] = action.get("quiz_params", {})
    return result


# ── Stage 6 — Evaluate ────────────────────────────────────────────────────────
def evaluate(session_id: str, latest_attempt: Dict) -> Dict:
    """
    Compare the latest attempt to the previous one for the same topics.
    Returns a dict of {topic: delta_pct, message}.
    """
    attempts = db.get_attempts(session_id, limit=3)
    if len(attempts) < 2:
        return {"message": "First attempt recorded. Keep going!", "deltas": {}}

    prev_id  = attempts[1]["id"]

    with db._conn() as con:
        prev_topics = {
            r["topic"]: r["pct"]
            for r in con.execute(
                "SELECT topic, pct FROM topic_scores WHERE attempt_id=?", (prev_id,)
            ).fetchall()
        }

    deltas: Dict[str, float] = {}
    messages: List[str] = []

    for t in latest_attempt.get("topic_chart", []):
        topic = t["topic"]
        curr  = t["pct"]
        prev  = prev_topics.get(topic)
        if prev is not None:
            delta = curr - prev
            deltas[topic] = delta
            if delta >= 10:
                messages.append(f"📈 {topic}: +{delta:.0f}% improvement!")
            elif delta <= -10:
                messages.append(f"📉 {topic}: {delta:.0f}% — needs attention.")

    return {
        "message": " &nbsp;·&nbsp; ".join(messages) if messages else "Scores stable.",
        "deltas":  deltas,
    }


# ── Stage 7 — Update ──────────────────────────────────────────────────────────
def update_plan(session_id: str, topic_chart: List[Dict]) -> None:
    """
    After a quiz attempt, update the spaced-repetition schedule for each topic.
    Uses score to determine SM-2 quality rating (0–5).
    """
    for t in topic_chart:
        pct   = t["pct"]
        topic = t["topic"]
        # Map percentage to SM-2 quality (0=complete blackout, 5=perfect)
        quality = int(min(5, max(0, round(pct / 20))))
        schedule_topic(session_id, topic, quality)


# ── Public entry point ────────────────────────────────────────────────────────
def run_coach(session_id: str, topics_text: Dict[str, str],
              latest_result: Optional[Dict] = None) -> Dict:
    """
    Full 7-stage agent loop.
    Call after every quiz submission or on dashboard load.
    Returns a complete coach response for the frontend.
    """
    # Stage 1
    obs = observe(session_id)
    obs["session_id"] = session_id

    # Stage 6+7 — if a new result just came in, record + evaluate first
    eval_result = {"message": "", "deltas": {}}
    if latest_result:
        eval_result = evaluate(session_id, latest_result)
        update_plan(session_id, latest_result.get("topic_chart", []))

    # Re-observe after update
    obs = observe(session_id)
    obs["session_id"] = session_id

    # Stages 2–5
    analysis = analyse(obs)
    plan     = build_plan(analysis, obs)
    action   = select_action(analysis, obs)
    executed = execute_action(action, topics_text)

    return {
        "insights":   analysis["insights"],
        "plan":       plan,
        "action":     executed,
        "evaluation": eval_result,
        "history": {
            "attempts":    len(obs["attempts"]),
            "topic_count": len(obs["topic_hist"]),
            "due_today":   len(obs["due_today"]),
        },
        "goal": obs["goal"],
    }
