"""
app.py  –  Brainora AI  (Flask backend)
Run:    python app.py
Visit:  http://127.0.0.1:5000
"""
import os, uuid, random, json
from datetime import date, datetime

from flask import Flask, render_template, request, session, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename

APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
APP_DATA_DIR = os.environ.get("APP_DATA_DIR") or os.path.dirname(__file__)

def _secret_key_for_environment(environment: str, secret_key: str | None) -> str:
    import secrets as _secrets
    if environment == "production" and not secret_key:
        # wsgi.py generates SECRET_KEY before importing this module.
        # If it's still missing (direct gunicorn app:app invocation), auto-generate.
        secret_key = _secrets.token_hex(32)
    return secret_key or "dev-only-insecure-key"

SECRET_KEY = _secret_key_for_environment(APP_ENV, os.environ.get("SECRET_KEY"))

from pdf_extractor     import extract_text, detect_topics
from question_generator import generate_mcqs, generate_quiz_from_topics
import database        as db
import agent_service   as coach
import revision_service as rev

# ── app ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY

SESSION_DIR = os.environ.get("SESSION_FILE_DIR") or os.path.join(APP_DATA_DIR, "flask_sessions")
os.makedirs(SESSION_DIR, exist_ok=True)
app.config.update(
    SESSION_TYPE      = "filesystem",
    SESSION_FILE_DIR  = SESSION_DIR,
    SESSION_PERMANENT = False,
    SESSION_COOKIE_HTTPONLY = True,
    SESSION_COOKIE_SAMESITE = "Lax",
    SESSION_COOKIE_SECURE = APP_ENV == "production",
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,
)
Session(app)

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER") or os.path.join(APP_DATA_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_PDF   = {".pdf"}


# ── helpers ───────────────────────────────────────────────────────────────────
def _sid() -> str:
    if "_sid" not in session:
        session["_sid"] = str(uuid.uuid4())
    return session["_sid"]

def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/dashboard")
def api_dashboard():
    sid      = _sid()
    attempts = db.get_attempts(sid, limit=50)
    th       = db.get_topic_history(sid)
    goal     = db.get_goal(sid)
    sched    = rev.get_schedule(sid)

    # topic mastery buckets
    strong = sum(1 for t in th if t["avg_pct"] >= 80)
    good   = sum(1 for t in th if 65 <= t["avg_pct"] < 80)
    needs  = sum(1 for t in th if 50 <= t["avg_pct"] < 65)
    weak   = sum(1 for t in th if t["avg_pct"] < 50)

    # overall %
    overall = round(sum(t["avg_pct"] for t in th) / len(th)) if th else 0

    # study streak (consecutive days with an attempt)
    streak = _calc_streak(attempts)

    # recent activity (last 4 events)
    recent = []
    for a in attempts[:4]:
        recent.append({
            "type":    "quiz",
            "title":   f"{a['pdf_name'] or 'Quiz'}",
            "detail":  f"{a['correct']}/{a['total']} correct",
            "time":    _relative_time(a["attempted_at"]),
            "tag":     "Weak Area" if a["score_pct"] < 60 else "Strong",
            "tag_cls": "tag-weak" if a["score_pct"] < 60 else "tag-strong",
        })

    # next exam
    exam_date = goal["exam_date"] if goal else None
    days_left = None
    if exam_date:
        try:
            d = date.fromisoformat(exam_date)
            days_left = (d - date.today()).days
        except Exception:
            pass

    return jsonify({
        "success":      True,
        "overall":      overall,
        "topics_total": len(th),
        "topics_mastered": strong + good,
        "total_quizzes": len(attempts),
        "streak":       streak,
        "exam_date":    exam_date,
        "days_left":    days_left,
        "mastery": {"strong": strong, "good": good, "needs": needs, "weak": weak},
        "recent":       recent,
        "due_today":    len(sched["due_today"]) + len(sched["overdue"]),
        "topic_history": th[:8],
    })


def _calc_streak(attempts):
    if not attempts:
        return 0
    from datetime import timedelta
    days = set()
    for a in attempts:
        try:
            days.add(datetime.fromisoformat(a["attempted_at"]).date())
        except Exception:
            pass
    streak, d = 0, date.today()
    while d in days:
        streak += 1
        d -= timedelta(days=1)
    return streak


def _relative_time(ts_str):
    try:
        ts  = datetime.fromisoformat(ts_str)
        now = datetime.now()
        diff = now - ts
        if diff.seconds < 3600:   return f"{diff.seconds//60}m ago"
        if diff.days == 0:        return f"{diff.seconds//3600}h ago"
        return f"{diff.days}d ago"
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD — PDF
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["pdf"]
    if _ext(file.filename or "") not in ALLOWED_PDF:
        return jsonify({"error": "Please upload a PDF file."}), 400

    fname    = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    fpath    = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    try:
        text = extract_text(fpath)
    finally:
        try: os.remove(fpath)
        except OSError: pass

    if len(text) < 200:
        return jsonify({"error": "Could not extract text. Use a text-based (non-scanned) PDF."}), 400

    topics = detect_topics(text)
    if len(topics) < 2:
        topics = {"General": text}

    session["raw_text"] = text
    session["topics"]   = {k: v for k, v in list(topics.items())[:15]}
    session["pdf_name"] = file.filename
    session.pop("questions",    None)
    session.pop("user_answers", None)

    # Save to file registry
    files = session.get("my_files", [])
    files.insert(0, {"name": file.filename, "type": "pdf",
                     "topics": len(topics), "date": date.today().isoformat()})
    session["my_files"] = files[:20]

    return jsonify({
        "success":    True,
        "pdf_name":   file.filename,
        "char_count": len(text),
        "topics":     list(topics.keys()),
    })


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE QUIZ
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/generate", methods=["POST"])
def api_generate():
    if "topics" not in session:
        return jsonify({"error": "NO_UPLOAD"}), 400

    data = request.get_json(silent=True) or {}
    num_q  = max(5, min(int(data.get("num_questions",       10)), 30))
    qpt    = max(1, min(int(data.get("questions_per_topic",  3)),  6))
    topic_filter = data.get("topic_filter")

    topics   = session["topics"]
    raw_text = session.get("raw_text", "")

    # Topic filter — generate only for the weak topic
    if topic_filter and topic_filter in topics:
        questions = generate_mcqs(topics[topic_filter], topic=topic_filter, num_questions=num_q)
    else:
        questions = generate_quiz_from_topics(topics, questions_per_topic=qpt, max_total=num_q+10)
        if len(questions) < num_q:
            extra = generate_mcqs(raw_text, topic="General",
                                  num_questions=num_q - len(questions) + 5)
            questions.extend(extra)

    random.shuffle(questions)
    questions = questions[:num_q]

    if not questions:
        return jsonify({"error": "TOO_SHORT"}), 400

    session["questions"] = questions
    return jsonify({
        "success":   True,
        "questions": [{"index": i, "topic": q["topic"],
                       "question": q["question"],
                       "options": q["options"],
                       "source": q.get("source", "")}
                      for i, q in enumerate(questions)],
        "total": len(questions),
    })


# ══════════════════════════════════════════════════════════════════════════════
# SUBMIT QUIZ
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/submit", methods=["POST"])
def api_submit():
    if "questions" not in session:
        return jsonify({"error": "No active quiz."}), 400

    data         = request.get_json(silent=True) or {}
    user_answers = data.get("answers", {})
    time_taken   = data.get("time_taken", {})
    questions    = session["questions"]
    total        = len(questions)
    correct_count = 0
    topic_stats   = {}
    detailed      = []

    for i, q in enumerate(questions):
        ua = user_answers.get(str(i), "")
        ok = ua == q["answer"]
        if ok: correct_count += 1
        t = q["topic"]
        if t not in topic_stats:
            topic_stats[t] = {"correct": 0, "total": 0}
        topic_stats[t]["total"] += 1
        if ok: topic_stats[t]["correct"] += 1
        detailed.append({
            "index": i+1, "question": q["question"], "topic": t,
            "options": q["options"], "user_ans": ua,
            "correct_ans": q["answer"], "answer_text": q["answer_text"],
            "is_correct": ok, "source": q.get("source", ""),
            "time_taken": time_taken.get(str(i), 0),
        })

    score_pct   = round(correct_count / total * 100) if total else 0
    topic_chart = [{"topic": t, "pct": round(s["correct"]/s["total"]*100),
                    "correct": s["correct"], "total": s["total"]}
                   for t, s in topic_stats.items()]
    weak   = [x for x in topic_chart if x["pct"] < 60]
    strong = [x for x in topic_chart if x["pct"] >= 60]

    session["user_answers"]  = user_answers
    session["last_results"]  = {"score_pct": score_pct,
                                 "correct": correct_count, "total": total}

    # Persist to DB + update coach
    sid = _sid()
    if total > 0:
        db.save_attempt(sid, session.get("pdf_name",""), score_pct,
                        correct_count, total, topic_chart)
    coach_resp = coach.run_coach(sid, session.get("topics",{}),
                                 {"topic_chart": topic_chart, "total": total})
    coach.update_plan(sid, topic_chart)

    return jsonify({
        "success":     True,
        "score_pct":   score_pct,
        "correct":     correct_count,
        "total":       total,
        "weak":        weak,
        "strong":      strong,
        "topic_chart": topic_chart,
        "detailed":    detailed,
        "pdf_name":    session.get("pdf_name",""),
        "coach":       {"action": coach_resp.get("action",{}),
                        "insights": coach_resp.get("insights",[])[:3]},
    })


# ══════════════════════════════════════════════════════════════════════════════
# STUDY COACH
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/coach")
def api_coach():
    sid  = _sid()
    resp = coach.run_coach(sid, session.get("topics", {}))
    return jsonify({"success": True, **resp})

@app.route("/api/coach/after-quiz", methods=["POST"])
def api_coach_after_quiz():
    sid   = _sid()
    data  = request.get_json(silent=True) or {}
    tc    = data.get("topic_chart", [])
    if data.get("total", 0) > 0:
        db.save_attempt(sid, session.get("pdf_name",""),
                        data.get("score_pct",0), data.get("correct",0),
                        data.get("total",0), tc)
    resp = coach.run_coach(sid, session.get("topics",{}), latest_result=data)
    return jsonify({"success": True, **resp})


# ══════════════════════════════════════════════════════════════════════════════
# AI TUTOR
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/tutor", methods=["POST"])
def api_tutor():
    """
    Retrieval-based tutor: finds the most relevant passage from the student's
    uploaded notes and returns it with a structured explanation.
    No LLM required — uses keyword overlap (TF-IDF style).
    """
    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please ask a question."}), 400

    raw_text = session.get("raw_text", "")
    topics   = session.get("topics", {})

    if not raw_text:
        return jsonify({
            "answer": "No study material uploaded yet. Please upload a PDF first.",
            "source": None, "confidence": 0,
        })

    passage, topic, score = _retrieve(question, topics)
    answer = _generate_answer(question, passage, topic)

    return jsonify({
        "success":    True,
        "answer":     answer,
        "source":     topic,
        "passage":    passage[:300] if passage else "",
        "confidence": score,
        "follow_up":  _follow_up_question(topic, topics.get(topic,"")),
    })


def _retrieve(question: str, topics: dict):
    """Simple keyword-overlap retrieval."""
    import re
    q_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
    best_score, best_topic, best_passage = 0, "General", ""
    for topic, text in topics.items():
        words = set(re.findall(r'\b\w{3,}\b', text.lower()))
        score = len(q_words & words) / max(len(q_words), 1)
        if score > best_score:
            best_score = score
            best_topic = topic
            # Find most relevant sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            best_sent = max(sentences,
                            key=lambda s: len(q_words & set(s.lower().split())),
                            default=text[:300])
            best_passage = best_sent[:600]
    return best_passage, best_topic, round(best_score * 100)


def _generate_answer(question: str, passage: str, topic: str) -> str:
    if not passage:
        return ("I couldn't find a relevant passage for your question in the "
                "uploaded material. Try rephrasing or upload more detailed notes.")
    return (f"Based on your notes on **{topic}**:\n\n"
            f"{passage}\n\n"
            f"*This answer is drawn directly from your uploaded study material.*")


def _follow_up_question(topic: str, text: str) -> str:
    if not text:
        return ""
    from question_generator import generate_mcqs
    qs = generate_mcqs(text, topic=topic, num_questions=1)
    if qs:
        q = qs[0]
        opts = "  ".join(f"{k}. {v}" for k, v in q["options"].items())
        return f"Practice Q: {q['question']}  |  {opts}"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# FLASHCARDS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/flashcards")
def api_flashcards():
    topics = session.get("topics", {})
    if not topics:
        return jsonify({"success": True, "flashcards": []})
    from question_generator import generate_mcqs
    cards = []
    for topic, text in list(topics.items())[:6]:
        qs = generate_mcqs(text, topic=topic, num_questions=2)
        for q in qs:
            cards.append({
                "front":  q["question"],
                "back":   f"{q['answer']}. {q['options'][q['answer']]}",
                "topic":  topic,
                "source": q.get("source","")[:120],
            })
    return jsonify({"success": True, "flashcards": cards})


# ══════════════════════════════════════════════════════════════════════════════
# STUDY RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/recommendations")
def api_recommendations():
    """
    Recommends review of uploaded notes for weak topics.
    """
    sid = _sid()
    th  = db.get_topic_history(sid)
    weak_topics = [t for t in th if t["avg_pct"] < 60]

    recs = []
    for t in weak_topics[:5]:
        recs.append({
            "topic":  t["topic"],
            "reason": f"You scored {t['avg_pct']}% on average — needs revision.",
            "action": "review_notes",
            "label":  "Review Notes",
            "source": "Your uploaded material",
        })

    if not recs:
        recs.append({
            "topic":  "All topics",
            "reason": "No weak topics found — great work! Keep practising to maintain mastery.",
            "action": "quiz",
            "label":  "Take a Practice Quiz",
            "source": "Your uploaded material",
        })

    return jsonify({"success": True, "recommendations": recs})


# ══════════════════════════════════════════════════════════════════════════════
# REVISION / GOALS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/revision")
def api_revision():
    return jsonify({"success": True, "schedule": rev.get_schedule(_sid())})

@app.route("/api/revision/complete", methods=["POST"])
def api_revision_complete():
    data = request.get_json(silent=True) or {}
    tid  = data.get("task_id")
    if not tid:
        return jsonify({"error": "task_id required"}), 400
    db.complete_revision_task(int(tid))
    return jsonify({"success": True, "schedule": rev.get_schedule(_sid())})

@app.route("/api/goal", methods=["POST"])
def api_goal():
    sid  = _sid()
    data = request.get_json(silent=True) or {}
    exam_date   = data.get("exam_date") or None
    daily_hours = float(data.get("daily_hours", 1.5))
    goal_text   = data.get("goal_text", "")
    db.save_goal(sid, exam_date, daily_hours, goal_text)
    if exam_date:
        th = db.get_topic_history(sid)
        if th:
            plan = rev.build_exam_plan(sid, exam_date, th)
            rev.save_exam_plan(sid, plan)
    return jsonify({"success": True, "goal": db.get_goal(sid)})


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY / ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/history")
def api_history():
    sid = _sid()
    return jsonify({
        "success":       True,
        "attempts":      db.get_attempts(sid),
        "topic_history": db.get_topic_history(sid),
    })

@app.route("/api/files")
def api_files():
    return jsonify({"success": True, "files": session.get("my_files", [])})


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=APP_ENV != "production",
            host="0.0.0.0" if APP_ENV == "production" else "127.0.0.1",
            port=port)
