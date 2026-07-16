import os
import re
import json
import base64
from datetime import datetime, timedelta
from typing import Optional
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for
import firebase_admin
from firebase_admin import credentials, auth, firestore as fb_firestore
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

cred = credentials.Certificate("firebase-service-account.json")
firebase_admin.initialize_app(cred)
db = fb_firestore.client()

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ================================================================
# FREE-TIER PYDANTIC MODELS
# ================================================================

class CareerRecommendation(BaseModel):
    title: str = Field(description="The job/career title.")
    fit_score: str = Field(description="Compatibility score, e.g., '95%'.")
    why_it_fits: str = Field(description="An explanation of why this matches the student's profile.")
    roadmap: list[str] = Field(description="A 3-step actionable roadmap to get started.")
    salary_range: Optional[str] = Field(default=None, description="Average salary range. Populate only if elite.")
    growth_forecast: Optional[str] = Field(default=None, description="10-year job growth. Populate only if elite.")
    certifications: Optional[list[str]] = Field(default=None, description="Key certifications. Populate only if elite.")
    interview_prep: Optional[list[str]] = Field(default=None, description="Interview questions. Populate only if elite.")


class CareerCounselingResponse(BaseModel):
    recommendations: list[CareerRecommendation]


# ================================================================
# ELITE COMPANION — CONSTANTS & HELPERS
# ================================================================

LEVEL_THRESHOLDS = [
    (0,    "Explorer"),
    (100,  "Seeker"),
    (300,  "Voyager"),
    (600,  "Visionary"),
    (1000, "Pioneer"),
]

DEFAULT_PROFILE = {
    "interests": [],
    "recent_events": [],
    "emotional_tone": "curious",
    "focus_score": 0,
    "xp": 0,
    "streak": 0,
    "last_checkin": None,
    "badges": [],
    "level": "Explorer",
    "summary": "Just getting started on their career journey.",
    "session_count": 0,
    "interest_scores": {
        "creative": 0, "technology": 0, "science": 0,
        "business": 0, "social": 0, "health": 0
    }
}


def get_level_name(xp: int) -> str:
    name = "Explorer"
    for threshold, n in LEVEL_THRESHOLDS:
        if xp >= threshold:
            name = n
    return name


def get_next_level(xp: int):
    """Return (threshold, name) for the next level, or (None, None) if maxed."""
    for threshold, name in LEVEL_THRESHOLDS:
        if xp < threshold:
            return threshold, name
    return None, None


def get_verified_user(req):
    """Verify the token cookie and return the decoded Firebase token. Raises on failure."""
    id_token = req.cookies.get("token")
    if not id_token:
        raise Exception("Not authenticated — no token cookie found.")
    return auth.verify_id_token(id_token, clock_skew_seconds=60)


def get_cookie_options() -> dict:
    """Return cookie options that work reliably in HTTPS deployments like Vercel."""
    is_secure = request.is_secure or "vercel.app" in request.host or "now.sh" in request.host
    return {"httponly": True, "secure": is_secure, "samesite": "Lax"}


def parse_token_metadata(token: str) -> dict:
    """Decode JWT header/payload without verification for debugging invalid signature issues."""
    parts = token.split('.')
    if len(parts) != 3:
        return {}

    def decode_part(part: str):
        try:
            padding = '=' * (-len(part) % 4)
            return json.loads(base64.urlsafe_b64decode(part + padding).decode('utf-8'))
        except Exception:
            return None

    header = decode_part(parts[0])
    payload = decode_part(parts[1])
    return {
        'iss': payload.get('iss') if isinstance(payload, dict) else None,
        'aud': payload.get('aud') if isinstance(payload, dict) else None,
        'sub': payload.get('sub') if isinstance(payload, dict) else None,
        'email': payload.get('email') if isinstance(payload, dict) else None,
        'header': header,
        'payload': payload,
    }


def get_profile(uid: str) -> dict:
    """Load the user's living profile from Firestore, with fallback to defaults."""
    try:
        doc = (
            db.collection("users").document(uid)
            .collection("profile").document("living_profile")
            .get()
        )
        if doc.exists:
            data = doc.to_dict()
            for key, default_val in DEFAULT_PROFILE.items():
                if key not in data:
                    data[key] = default_val
            return data
    except Exception:
        pass
    return dict(DEFAULT_PROFILE)


def save_profile(uid: str, profile: dict):
    """Persist the user's profile to Firestore."""
    db.collection("users").document(uid).collection("profile").document("living_profile").set(profile)


def get_message_history(uid: str, limit: int = 30) -> list:
    """
    Return the last N messages for a user, oldest first.
    Each item is {"role": "user"|"model", "content": "..."}
    """
    try:
        docs = (
            db.collection("users").document(uid)
            .collection("messages")
            .order_by("timestamp", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        msgs = [{"role": d.to_dict().get("role", "user"), "content": d.to_dict().get("content", "")}
                for d in docs]
        return list(reversed(msgs))
    except Exception:
        return []


def save_message(uid: str, role: str, content: str):
    """Save a single chat message to Firestore."""
    db.collection("users").document(uid).collection("messages").add({
        "role": role,
        "content": content,
        "timestamp": fb_firestore.SERVER_TIMESTAMP,
    })


def build_system_prompt(profile: dict, email: str) -> str:
    """
    Build a rich, personalised system instruction from the user's living profile.
    The instruction instructs Gemini to append a hidden JSON profile-update block
    (<<<JSON>>>...<<<END>>>) after every visible response.
    """
    interests = ", ".join(profile.get("interests", [])) or "not yet identified"
    events    = "; ".join(profile.get("recent_events", [])) or "nothing shared yet"

    return (
        f"You are a warm, perceptive, and proactive AI career companion for {email}. "
        f"You are NOT a generic Q&A bot. You have a persistent, evolving memory of this specific person.\n\n"

        f"=== WHAT YOU KNOW ABOUT THEM ===\n"
        f"Interests: {interests}\n"
        f"Recent life events they shared: {events}\n"
        f"Emotional tone lately: {profile.get('emotional_tone', 'neutral')}\n"
        f"Career focus score: {profile.get('focus_score', 0)}/100\n"
        f"Summary: {profile.get('summary', 'Just getting started.')}\n\n"

        f"=== YOUR BEHAVIOUR ===\n"
        f"- Always reference what you know about them naturally, like a friend who remembers.\n"
        f"- Ask exactly ONE thoughtful follow-up question per response.\n"
        f"- Notice emerging patterns in their interests and name them explicitly.\n"
        f"- Be warm, direct, and concise. NOT formal or report-like.\n"
        f"- Check in on events they mentioned in previous sessions.\n"
        f"- Responses should be 2-5 sentences of visible text unless they ask for depth.\n\n"

        f"=== MANDATORY PROFILE UPDATE BLOCK ===\n"
        f"After EVERY response, append the following block. It is NEVER shown to the user — "
        f"it is extracted by the system for profile storage. Do NOT mention or reference this block in your reply.\n\n"
        f"<<<JSON>>>\n"
        f"{{\n"
        f"  \"interests\": [up to 5 interests, updated from full conversation],\n"
        f"  \"recent_events\": [up to 5 significant events mentioned, updated],\n"
        f"  \"emotional_tone\": \"one descriptive phrase\",\n"
        f"  \"focus_score\": integer 0-100 (how clearly defined is their career direction),\n"
        f"  \"summary\": \"1-2 sentences summarising this person's current career self-discovery state\",\n"
        f"  \"interest_scores\": {{\"creative\": 0-100, \"technology\": 0-100, \"science\": 0-100, "
        f"\"business\": 0-100, \"social\": 0-100, \"health\": 0-100}},\n"
        f"  \"new_badge\": \"badge_key or null. Options: 'life_update' if they shared a personal event, "
        f"'focus_locked' if focus_score >= 70\"\n"
        f"}}\n"
        f"<<<END>>>"
    )


def parse_ai_response(raw_text: str):
    """
    Split the raw AI response into:
    - visible_text: the message shown to the user
    - profile_updates: dict parsed from the hidden JSON block, or None
    """
    match = re.search(r'<<<JSON>>>(.*?)<<<END>>>', raw_text, re.DOTALL)
    if match:
        visible = raw_text[:match.start()].strip()
        try:
            updates = json.loads(match.group(1).strip())
            return visible, updates
        except json.JSONDecodeError:
            return raw_text.strip(), None
    return raw_text.strip(), None


def format_quota_error_message(exc: Exception) -> Optional[str]:
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        retry_match = re.search(r'retryDelay[^0-9]*([0-9]+(?:\.[0-9]+)?)s', text)
        if not retry_match:
            retry_match = re.search(r'(\d+)\s*second', text)
        if retry_match:
            seconds = float(retry_match.group(1))
            reset_time = datetime.utcnow() + timedelta(seconds=seconds)
            return (
                f"Token limit has been exceeded. Please try later "
                f"({reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')})."
            )
        return "Token limit has been exceeded. Please try later."
    return None


def update_gamification(profile: dict, xp_to_add: int) -> dict:
    """
    Award XP, update level name, update daily streak.
    Returns the mutated profile.
    """
    from datetime import date

    today = date.today().isoformat()
    last_checkin = profile.get("last_checkin")

    # Streak logic
    if last_checkin:
        try:
            last_date = date.fromisoformat(str(last_checkin)[:10])
            diff = (date.today() - last_date).days
            if diff == 1:
                profile["streak"] = profile.get("streak", 0) + 1
            elif diff > 1:
                profile["streak"] = 1
            # diff == 0 → same day, no change
        except ValueError:
            profile["streak"] = 1
    else:
        profile["streak"] = 1

    profile["last_checkin"] = today

    # XP + level
    profile["xp"] = profile.get("xp", 0) + xp_to_add
    profile["level"] = get_level_name(profile["xp"])

    # Streak badges
    streak = profile.get("streak", 0)
    badges = profile.setdefault("badges", [])
    if streak >= 7 and "streak_7" not in badges:
        badges.append("streak_7")
    elif streak >= 3 and "streak_3" not in badges:
        badges.append("streak_3")

    return profile


# ================================================================
# EXISTING FREE-TIER ROUTES
# ================================================================

@app.route("/", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/set-cookie", methods=["POST"])
def set_cookie():
    """Receives the Firebase ID token, verifies it, and sets an httpOnly session cookie."""
    data = request.get_json()
    id_token = data.get("idToken")
    try:
        decoded = auth.verify_id_token(id_token, clock_skew_seconds=60)
        resp = make_response(jsonify({"status": "success", "premium": decoded.get("premium", False)}))
        resp.set_cookie("token", id_token, **get_cookie_options())
        return resp
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 401


@app.route("/dashboard", methods=["GET"])
def dashboard():
    id_token = request.cookies.get("token")
    if not id_token:
        return redirect(url_for("login_page"))
    try:
        user_info = auth.verify_id_token(id_token, clock_skew_seconds=60)
        is_premium = user_info.get("premium", False)
        email = user_info.get("email", "")
        return render_template(
            "index.html",
            academic_background="", favorite_subjects="",
            hobbies_interests="", skills="",
            work_env="Remote / Tech-focused office",
            is_premium=is_premium, email=email,
        )
    except Exception:
        return redirect(url_for("login_page"))


@app.route("/upgrade", methods=["POST"])
def upgrade():
    data = request.get_json(silent=True) or {}
    id_token = data.get("idToken")
    cookie_token = request.cookies.get("token")

    if not id_token and not cookie_token:
        return jsonify({"status": "error", "message": "Unauthorized: token not provided."}), 401

    def verify_token(token):
        return auth.verify_id_token(token, clock_skew_seconds=60)

    def debug_token(token: str) -> dict:
        if not token:
            return {}
        meta = parse_token_metadata(token)
        return {
            "token": token[:32] + "...",
            "metadata": meta,
            "length": len(token),
        }

    try:
        user_info = verify_token(id_token) if id_token else verify_token(cookie_token)
    except Exception as body_exc:
        if cookie_token and id_token and cookie_token != id_token:
            try:
                user_info = verify_token(cookie_token)
            except Exception as cookie_exc:
                return jsonify({
                    "status": "error",
                    "message": f"Upgrade failed: body token error: {body_exc}; cookie token error: {cookie_exc}",
                    "debug": {
                        "body_token": debug_token(id_token),
                        "cookie_token": debug_token(cookie_token),
                    },
                }), 401
        else:
            return jsonify({
                "status": "error",
                "message": f"Upgrade failed: {str(body_exc)}",
                "debug": {"token": debug_token(id_token or cookie_token)},
            }), 401

    try:
        uid = user_info["uid"]
        auth.set_custom_user_claims(uid, {"premium": True})
        auth.revoke_refresh_tokens(uid)
        return jsonify({"status": "success", "premium": True})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Upgrade failed: {str(e)}"}), 401


@app.route("/logout", methods=["POST"])
def logout():
    resp = make_response(redirect(url_for("login_page")))
    resp.set_cookie("token", "", expires=0)
    return resp


@app.route("/recommend", methods=["POST"])
def recommend():
    id_token = request.cookies.get("token")
    if not id_token:
        return redirect(url_for("login_page"))
    try:
        user_info = auth.verify_id_token(id_token, clock_skew_seconds=60)
        is_premium = user_info.get("premium", False)
        email = user_info.get("email", "")
    except Exception:
        return redirect(url_for("login_page"))

    academic_background = request.form.get("academic_background", "").strip()
    favorite_subjects   = request.form.get("favorite_subjects",   "").strip()
    hobbies_interests   = request.form.get("hobbies_interests",   "").strip()
    skills              = request.form.get("skills",              "").strip()
    work_env            = request.form.get("work_env", "Remote / Tech-focused office")

    base_ctx = dict(
        academic_background=academic_background, favorite_subjects=favorite_subjects,
        hobbies_interests=hobbies_interests, skills=skills,
        work_env=work_env, is_premium=is_premium, email=email,
    )

    if not academic_background or not favorite_subjects:
        return render_template("index.html", error="Please fill in at least Academic Background and Favorite Subjects.", **base_ctx), 400

    student_profile = {
        "academic_background": academic_background,
        "favorite_subjects":   [s.strip() for s in favorite_subjects.split(",") if s.strip()],
        "hobbies_interests":   [h.strip() for h in hobbies_interests.split(",") if h.strip()],
        "skills":              [sk.strip() for sk in skills.split(",") if sk.strip()],
        "preferred_work_environment": work_env,
    }

    system_instruction = (
        "You are an expert, empathetic AI Career Counselor. Help the user decide their career path. "
        + ("Since the user is an ELITE member, you MUST populate all premium fields: salary_range, "
           "growth_forecast, certifications, and interview_prep."
           if is_premium else
           "The user is on the free tier. Do NOT populate salary_range, growth_forecast, "
           "certifications, or interview_prep — leave them null.")
    )

    if gemini_client is None:
        return render_template(
            "index.html",
            error="The Gemini API key is not configured. Please add GEMINI_API_KEY in your Vercel project settings.",
            **base_ctx,
        ), 500

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Recommend 3 careers for this student:\n\n{student_profile}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=CareerCounselingResponse,
                temperature=0.7,
            ),
        )
        result = resp.parsed
        return render_template("index.html", recommendations=result.recommendations,
                               student_profile=student_profile, **base_ctx)
    except Exception as exc:
        quota_msg = format_quota_error_message(exc)
        if quota_msg:
            return render_template("index.html", error=quota_msg, **base_ctx), 500
        return render_template("index.html", error=f"An error occurred: {exc}", **base_ctx), 500


# ================================================================
# ELITE COMPANION ROUTES
# ================================================================

@app.route("/elite", methods=["GET"])
def elite():
    """Serve the Elite Companion page (premium users only)."""
    id_token = request.cookies.get("token")
    if not id_token:
        return redirect(url_for("login_page"))
    try:
        user_info = auth.verify_id_token(id_token, clock_skew_seconds=60)
        if not user_info.get("premium", False):
            return redirect(url_for("dashboard"))
        return render_template("premium.html", email=user_info.get("email", ""))
    except Exception:
        return redirect(url_for("login_page"))


@app.route("/api/elite/profile", methods=["GET"])
def api_elite_profile():
    """Return the user's full living profile."""
    try:
        user_info = get_verified_user(request)
        if not user_info.get("premium", False):
            return jsonify({"error": "Premium required"}), 403
        uid = user_info["uid"]
        profile = get_profile(uid)
        next_xp, next_level = get_next_level(profile.get("xp", 0))
        return jsonify({**profile, "next_level_xp": next_xp, "next_level_name": next_level})
    except Exception as e:
        return jsonify({"error": str(e)}), 401


@app.route("/api/elite/history", methods=["GET"])
def api_elite_history():
    """Return the last 50 chat messages for the user."""
    try:
        user_info = get_verified_user(request)
        if not user_info.get("premium", False):
            return jsonify({"error": "Premium required"}), 403
        messages = get_message_history(user_info["uid"], limit=50)
        return jsonify({"messages": messages})
    except Exception as e:
        return jsonify({"error": str(e)}), 401


@app.route("/api/elite/checkin", methods=["GET"])
def api_elite_checkin():
    """
    Generate a proactive, personalized greeting for the user on session start.
    Awards +25 XP and updates streak.
    """
    try:
        user_info = get_verified_user(request)
        if not user_info.get("premium", False):
            return jsonify({"error": "Premium required"}), 403

        uid   = user_info["uid"]
        email = user_info.get("email", "")
        profile = get_profile(uid)
        is_first = profile.get("session_count", 0) == 0

        if is_first:
            greeting_prompt = (
                "You are an AI career companion meeting a student for the first time. "
                "Give a warm, exciting, personalised welcome in 3-4 sentences. "
                "Explain that you will learn about them across many conversations, track their evolving interests, "
                "and help them find their direction. End with ONE opening question about what excites them most right now."
            )
        else:
            interests = ", ".join(profile.get("interests", [])) or "various topics"
            events    = "; ".join(profile.get("recent_events", [])) or "nothing yet"
            greeting_prompt = (
                f"You are an AI career companion reconnecting with a student you know well. "
                f"Their interests: {interests}. Recent events they shared: {events}. "
                f"Focus score: {profile.get('focus_score', 0)}/100. "
                f"Write a warm, specific 2-3 sentence greeting that references something concrete you know about them. "
                f"Then ask ONE focused follow-up question that helps deepen their self-understanding."
            )

        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=greeting_prompt,
            config=types.GenerateContentConfig(temperature=0.85),
        )
        greeting = resp.text.strip()

        # Update profile
        profile["session_count"] = profile.get("session_count", 0) + 1
        new_badges: list[str] = []

        if is_first and "first_session" not in profile.get("badges", []):
            profile.setdefault("badges", []).append("first_session")
            new_badges.append("first_session")

        profile = update_gamification(profile, xp_to_add=25)

        if profile.get("session_count", 0) >= 10 and "deep_diver" not in profile.get("badges", []):
            profile["badges"].append("deep_diver")
            new_badges.append("deep_diver")

        save_profile(uid, profile)
        save_message(uid, "model", greeting)

        next_xp, next_level = get_next_level(profile.get("xp", 0))
        return jsonify({
            "greeting": greeting,
            "profile": profile,
            "new_badges": new_badges,
            "next_level_xp": next_xp,
            "next_level_name": next_level,
        })

    except Exception as e:
        quota_msg = format_quota_error_message(e)
        if quota_msg:
            return jsonify({"error": quota_msg}), 500
        return jsonify({"error": str(e)}), 500


@app.route("/api/elite/chat", methods=["POST"])
def api_elite_chat():
    """
    Handle a chat message from the user.
    - Loads conversation history from Firestore
    - Calls Gemini with full history + personalised system prompt
    - Parses the hidden JSON profile-update block from the response
    - Saves messages, updates profile, awards XP
    - Returns visible message + updated gamification stats
    """
    try:
        user_info = get_verified_user(request)
        if not user_info.get("premium", False):
            return jsonify({"error": "Premium required"}), 403

        uid   = user_info["uid"]
        email = user_info.get("email", "")
        data  = request.get_json()
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        profile = get_profile(uid)
        history = get_message_history(uid, limit=20)

        # Build contents list for Gemini multi-turn
        contents = [
            types.Content(role=msg["role"], parts=[types.Part(text=msg["content"])])
            for msg in history
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(profile, email),
                temperature=0.8,
            ),
        )

        raw_text = resp.text
        visible_text, profile_updates = parse_ai_response(raw_text)

        # Persist both messages
        save_message(uid, "user",  user_message)
        save_message(uid, "model", visible_text)

        # Apply profile updates from AI
        if profile_updates:
            for key in ["interests", "recent_events", "emotional_tone", "focus_score", "summary", "interest_scores"]:
                if key in profile_updates:
                    profile[key] = profile_updates[key]

        # Determine XP reward
        life_keywords = [
            "exam", "test", "club", "joined", "quit", "failed", "passed",
            "family", "friend", "moved", "school", "won", "lost", "changed",
        ]
        xp_earned = 10
        new_badges: list[str] = []

        if any(kw in user_message.lower() for kw in life_keywords):
            xp_earned += 25
            if "life_update" not in profile.get("badges", []):
                new_badges.append("life_update")

        if profile_updates and profile_updates.get("new_badge"):
            badge = profile_updates["new_badge"]
            if isinstance(badge, str) and badge not in profile.get("badges", []):
                new_badges.append(badge)

        if profile.get("focus_score", 0) >= 70 and "focus_locked" not in profile.get("badges", []):
            new_badges.append("focus_locked")

        profile = update_gamification(profile, xp_to_add=xp_earned)

        for badge in new_badges:
            if badge not in profile.get("badges", []):
                profile["badges"].append(badge)

        save_profile(uid, profile)

        next_xp, next_level = get_next_level(profile.get("xp", 0))

        return jsonify({
            "message": visible_text,
            "profile": {
                "xp":             profile.get("xp", 0),
                "level":          profile.get("level", "Explorer"),
                "focus_score":    profile.get("focus_score", 0),
                "streak":         profile.get("streak", 0),
                "badges":         profile.get("badges", []),
                "interests":      profile.get("interests", []),
                "interest_scores": profile.get("interest_scores", {}),
                "summary":        profile.get("summary", ""),
            },
            "new_badges":     new_badges,
            "xp_earned":      xp_earned,
            "next_level_xp":  next_xp,
            "next_level_name": next_level,
        })

    except Exception as e:
        quota_msg = format_quota_error_message(e)
        if quota_msg:
            return jsonify({"error": quota_msg}), 500
        return jsonify({"error": str(e)}), 500


# ================================================================
if __name__ == "__main__":
    app.run()