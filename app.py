"""
app.py  —  Housing Affordability Intelligence Dashboard
=======================================================
Dual-mode backend:
  • SUPABASE mode  : when SUPABASE_URL + SUPABASE_SERVICE_KEY are set
                     → users, chart data, and model files come from Supabase
  • LOCAL mode     : falls back to local CSV / pkl files (development)
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
import os, io, json, hashlib, secrets, re
import numpy as np
import joblib
from functools import wraps
from datetime import timedelta

# ── optional dotenv for local dev ────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key               = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(days=7)

# ── Supabase client (optional) ────────────────────────────────
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
USE_SUPABASE         = bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)
sb                   = None

if USE_SUPABASE:
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print(f"[INFO] Supabase connected: {SUPABASE_URL}")
    except Exception as e:
        print(f"[WARN] Supabase init failed: {e} — falling back to local mode")
        USE_SUPABASE = False
else:
    print("[INFO] No Supabase credentials found — running in LOCAL mode")

# ── File paths (local fallback) ───────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(BASE_DIR, "refined_metadata.csv")
ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "cluster_assignments.csv")
MODEL_PATH       = os.path.join(BASE_DIR, "kmeans_model.pkl")
SCALER_PATH      = os.path.join(BASE_DIR, "scaler.pkl")
USERS_FILE       = os.path.join(BASE_DIR, "users.json")

# ── Global state ─────────────────────────────────────────────
model         = None
scaler        = None
cluster_stats = {}    # {0: {zinc2, costmed, cost_burden_ratio, age1, bedrms, Count}, …}
scatter_cache = []    # list of dicts for the scatter plot


# ═════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════

def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _to_python(obj):
    """Recursively convert numpy types → plain Python (JSON-safe)."""
    if isinstance(obj, dict):
        return {_to_python(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═════════════════════════════════════════════════════════════
#  AUTH — USER STORE
#  Uses Supabase `users` table when available, else users.json
# ═════════════════════════════════════════════════════════════

def _get_user(username: str):
    """Return user dict or None."""
    if USE_SUPABASE:
        res = sb.table("users").select("*").eq("username", username).execute()
        rows = res.data
        return rows[0] if rows else None
    # --- local fallback ---
    users = _local_load_users()
    u = users.get(username)
    if u:
        u["username"] = username
    return u


def _create_user(username: str, full_name: str, email: str, password: str):
    """Insert a new user. Returns (True, None) or (False, error_str)."""
    if USE_SUPABASE:
        try:
            sb.table("users").insert({
                "username":      username,
                "full_name":     full_name,
                "email":         email,
                "password_hash": _hash_pw(password),
            }).execute()
            return True, None
        except Exception as e:
            err = str(e)
            if "duplicate" in err.lower() or "unique" in err.lower():
                return False, f'Username "{username}" is already taken.'
            return False, err
    # --- local fallback ---
    users = _local_load_users()
    if username in users:
        return False, f'Username "{username}" is already taken.'
    users[username] = {
        "full_name":     full_name,
        "email":         email,
        "password_hash": _hash_pw(password),
    }
    _local_save_users(users)
    return True, None


def _local_load_users():
    if not os.path.exists(USERS_FILE):
        default = {"admin": {
            "full_name":     "Admin User",
            "email":         "admin@housing.ai",
            "password_hash": _hash_pw("housing123"),
        }}
        _local_save_users(default)
        return default
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _local_save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ═════════════════════════════════════════════════════════════
#  RESOURCE LOADING
# ═════════════════════════════════════════════════════════════

def load_resources():
    global model, scaler, cluster_stats, scatter_cache

    if USE_SUPABASE:
        _load_from_supabase()
    else:
        _load_from_local()


def _load_from_supabase():
    global model, scaler, cluster_stats, scatter_cache
    print("[INFO] Loading resources from Supabase …")

    # ── cluster_stats ─────────────────────────────────────────
    try:
        res = sb.table("cluster_stats").select("*").execute()
        for row in res.data:
            cid = row["cluster_id"]
            cluster_stats[cid] = {
                "ZINC2":             row["zinc2"],
                "COSTMED":           row["costmed"],
                "cost_burden_ratio": row["cost_burden_ratio"],
                "AGE1":              row["age1"],
                "BEDRMS":            row["bedrms"],
                "Count":             row["count"],
            }
        print(f"[INFO] cluster_stats loaded ({len(cluster_stats)} clusters)")
    except Exception as e:
        print(f"[WARN] Could not load cluster_stats: {e}")

    # ── scatter_data ──────────────────────────────────────────
    try:
        res = sb.table("scatter_data").select("zinc2,cost_burden_ratio,cluster_label,costmed").execute()
        scatter_cache = [
            {
                "ZINC2":             r["zinc2"],
                "cost_burden_ratio": r["cost_burden_ratio"],
                "Cluster_Label":     r["cluster_label"],
                "COSTMED":           r["costmed"],
            }
            for r in res.data
        ]
        print(f"[INFO] scatter_data loaded ({len(scatter_cache)} rows)")
    except Exception as e:
        print(f"[WARN] Could not load scatter_data: {e}")

    # ── model files from Storage ──────────────────────────────
    for attr, fname, path in [
        ("model",  "kmeans_model.pkl", MODEL_PATH),
        ("scaler", "scaler.pkl",       SCALER_PATH),
    ]:
        try:
            raw = sb.storage.from_("models").download(fname)
            obj = joblib.load(io.BytesIO(raw))
            globals()[attr] = obj
            print(f"[INFO] {fname} loaded from Supabase Storage")
        except Exception as e:
            print(f"[WARN] Could not load {fname} from Storage: {e}")
            # Try local fallback
            if os.path.exists(path):
                globals()[attr] = joblib.load(path)
                print(f"[INFO] {fname} loaded from local file (fallback)")


def _load_from_local():
    global model, scaler, cluster_stats, scatter_cache
    print("[INFO] Loading resources from local files …")

    try:
        import pandas as pd
        if os.path.exists(DATA_PATH) and os.path.exists(ASSIGNMENTS_PATH):
            metadata    = pd.read_csv(DATA_PATH)
            assignments = pd.read_csv(ASSIGNMENTS_PATH)
            if "Cluster_Label" in assignments.columns:
                metadata["Cluster_Label"] = assignments["Cluster_Label"]

            cols  = [c for c in ["ZINC2","COSTMED","cost_burden_ratio","AGE1","BEDRMS"] if c in metadata.columns]
            stats = metadata.groupby("Cluster_Label")[cols].mean()
            stats["Count"] = metadata["Cluster_Label"].value_counts()
            cluster_stats  = _to_python(stats.to_dict(orient="index"))

            sample       = metadata.sample(n=min(2000, len(metadata)), random_state=42).fillna(0)
            scatter_cache = _to_python(
                sample[["ZINC2","cost_burden_ratio","Cluster_Label","COSTMED"]].to_dict(orient="records")
            )
            print(f"[INFO] Local CSV loaded ({len(metadata)} rows)")
    except Exception as e:
        print(f"[WARN] Could not load local CSV: {e}")

    if os.path.exists(MODEL_PATH):
        model  = joblib.load(MODEL_PATH)
        print("[INFO] kmeans_model.pkl loaded locally")
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("[INFO] scaler.pkl loaded locally")


# Kick off on startup
load_resources()


# ═════════════════════════════════════════════════════════════
#  DECORATORS
# ═════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Authentication required", "redirect": "/login"}), 401
        return f(*args, **kwargs)
    return decorated


# ═════════════════════════════════════════════════════════════
#  ROUTES — MAIN
# ═════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        username  = session.get("username"),
        full_name = session.get("full_name", session.get("username")),
    )


@app.route("/health")
def health():
    return jsonify({
        "status":       "healthy" if model is not None else "degraded",
        "mode":         "supabase" if USE_SUPABASE else "local",
        "model_loaded": model is not None,
        "scatter_rows": len(scatter_cache),
        "clusters":     len(cluster_stats),
    })


# ═════════════════════════════════════════════════════════════
#  ROUTES — AUTH
# ═════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET"])
def login_page():
    if "username" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():
    data     = request.get_json() if request.is_json else request.form
    username = (data.get("username") or "").strip()
    password =  data.get("password") or ""
    remember = bool(data.get("remember", False))

    if not username or not password:
        err = "Username and password are required."
        if request.is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(url_for("login_page"))

    user = _get_user(username)
    if not user or user.get("password_hash") != _hash_pw(password):
        err = "Invalid username or password."
        if request.is_json:
            return jsonify({"success": False, "error": err}), 401
        flash(err, "error")
        return redirect(url_for("login_page"))

    if remember:
        session.permanent = True
    session["username"]  = username
    session["full_name"] = user.get("full_name", username)

    if request.is_json:
        return jsonify({"success": True, "redirect": url_for("index")})
    return redirect(url_for("index"))


@app.route("/register", methods=["GET"])
def register_page():
    if "username" in session:
        return redirect(url_for("index"))
    return render_template("register.html")


@app.route("/register", methods=["POST"])
def register_post():
    data             = request.get_json() if request.is_json else request.form
    full_name        = (data.get("full_name") or "").strip()
    username         = (data.get("username") or "").strip()
    email            = (data.get("email") or "").strip()
    password         = data.get("password") or ""
    confirm_password = data.get("confirm_password") or ""

    # Validation
    if not full_name or not username or not password or not confirm_password:
        err = "All fields are required."
        if request.is_json: return jsonify({"success": False, "error": err}), 400
        flash(err, "error"); return redirect(url_for("register_page"))

    if not re.match(r"^[a-zA-Z0-9_]{3,20}$", username):
        err = "Username: 3–20 characters, letters/numbers/underscores only."
        if request.is_json: return jsonify({"success": False, "error": err}), 400
        flash(err, "error"); return redirect(url_for("register_page"))

    if len(password) < 6:
        err = "Password must be at least 6 characters."
        if request.is_json: return jsonify({"success": False, "error": err}), 400
        flash(err, "error"); return redirect(url_for("register_page"))

    if password != confirm_password:
        err = "Passwords do not match."
        if request.is_json: return jsonify({"success": False, "error": err}), 400
        flash(err, "error"); return redirect(url_for("register_page"))

    ok, err = _create_user(username, full_name, email, password)
    if not ok:
        if request.is_json: return jsonify({"success": False, "error": err}), 409
        flash(err, "error"); return redirect(url_for("register_page"))

    if request.is_json:
        return jsonify({"success": True, "message": "Account created!", "redirect": url_for("login_page")})
    flash("Account created! Please sign in.", "success")
    return redirect(url_for("login_page"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login_page"))


# ═════════════════════════════════════════════════════════════
#  ROUTES — API (data / predict / clusters)
# ═════════════════════════════════════════════════════════════

@app.route("/api/data")
@api_login_required
def get_data():
    try:
        if not cluster_stats:
            return jsonify({"error": "Data not loaded"}), 503
        return jsonify({"scatter": scatter_cache, "stats": cluster_stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clusters")
@api_login_required
def get_clusters():
    return jsonify([
        {"id": 0, "label": "Middle-Income Stable",          "color": "#38bdf8", "icon": "🏠",
         "policy": "Workforce housing support, rent stabilization, first-time homebuyer assistance."},
        {"id": 1, "label": "Low-Income Severely Burdened",  "color": "#ef4444", "icon": "⚠️",
         "policy": "Immediate rent subsidies, eviction prevention, emergency housing assistance."},
        {"id": 2, "label": "High-Income Secure",            "color": "#22c55e", "icon": "✅",
         "policy": "Market-rate development, inclusionary zoning, cross-subsidy for affordable housing."},
        {"id": 3, "label": "Extremely Low-Income / No Income", "color": "#f59e0b", "icon": "🆘",
         "policy": "Emergency shelter, supportive housing, income verification and benefits enrollment."},
    ])


@app.route("/api/predict", methods=["POST"])
@api_login_required
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data           = request.json
        income_input   = float(data.get("income",   50000))
        cost_input     = float(data.get("cost",      1000))
        bedrooms_input = float(data.get("bedrooms",     3))
        raw_age        = data.get("age")

        # ── Age validation ────────────────────────────────────
        AGE_MEDIAN  = 44.0
        final_age   = AGE_MEDIAN
        age_anomaly = False

        if raw_age is None or raw_age == "":
            print("LOG: age_defaulted_missing")
        else:
            try:
                parsed = float(raw_age)
                if parsed < 18 or parsed > 110:
                    age_anomaly = True
                    print("LOG: age_out_of_range")
                else:
                    final_age = parsed
            except ValueError:
                print("LOG: age_defaulted_missing")

        # ── Feature builder ───────────────────────────────────
        def build_features(inc, cst, ag, beds):
            burden    = (cst * 12) / max(inc, 1)
            val_est   = cst * 12 * 15
            fmr_est   = 1100
            rooms_est = beds + 2
            status    = 1 if burden < 0.3 else (2 if burden <= 0.5 else 3)
            ami_ratio = inc / 70000.0
            ami_cat   = 1 if ami_ratio < 0.3 else (2 if ami_ratio < 0.5 else (3 if ami_ratio < 0.8 else 4))
            rent_own  = struct_type = 1 if inc > 65000 else 2
            return [inc, cst, val_est, fmr_est, burden, status, ag, 2, beds,
                    rooms_est, struct_type, rent_own, 3, 1, ami_cat], burden

        safe_income = float(income_input)
        safe_cost   = float(cost_input)
        safe_age    = float(final_age)
        safe_beds   = float(bedrooms_input)

        primary_vec, primary_burden = build_features(safe_income, safe_cost, safe_age, safe_beds)
        primary_scaled  = scaler.transform([primary_vec])
        primary_cluster = int(model.predict(primary_scaled)[0])

        # ── Policy map ────────────────────────────────────────
        POLICY = {
            0: ("Middle-Income Stable",           "Workforce housing support, rent stabilization, first-time homebuyer assistance."),
            1: ("Low-Income Severely Burdened",   "Immediate rent subsidies, eviction prevention, emergency housing assistance."),
            2: ("High-Income Secure",             "Market-rate development, inclusionary zoning, cross-subsidy for affordable housing."),
            3: ("Extremely Low-Income / No Income","Emergency shelter, supportive housing, income verification and benefits enrollment."),
        }
        if primary_cluster not in POLICY:
            return jsonify({"error": f"Invalid cluster: {primary_cluster}"}), 500

        label_str, policy_str = POLICY[primary_cluster]
        recommendation        = f"{label_str}: {policy_str}"

        # ── Anomaly detection ─────────────────────────────────
        possible_monthly = (safe_income < 6000) and (safe_cost >= 800) and (primary_burden > 2.0)
        anomalies = []
        if age_anomaly:
            anomalies.append(f"Age {raw_age} is out of valid range (18–110); median used.")
        if possible_monthly:
            anomalies.append("Potential monthly income entry detected.")

        anomaly_flag = bool(anomalies)
        burden_label = "High" if primary_burden > 0.3 else "Affordable"

        # ── Alternative (monthly income correction) ───────────
        alternative = None
        if possible_monthly:
            alt_inc  = safe_income * 12
            alt_vec, alt_burden = build_features(alt_inc, safe_cost, safe_age, safe_beds)
            alt_cl   = int(model.predict(scaler.transform([alt_vec]))[0])
            alt_lbl, alt_pol = POLICY.get(alt_cl, ("Unknown", ""))
            alternative = {
                "adjusted_annual_income": alt_inc,
                "predicted_cluster": {"id": alt_cl, "label": alt_lbl},
                "cost_burden_ratio":  alt_burden,
                "recommendation":     f"{alt_lbl}: {alt_pol}",
            }

        # ── Notes ─────────────────────────────────────────────
        notes = []
        if raw_age is None or raw_age == "":
            notes.append("age_defaulted_median")
        elif age_anomaly:
            notes.append(f"age_out_of_valid_range: {float(raw_age)}")

        return jsonify({
            # Nested (tests)
            "primary_result": {
                "predicted_cluster": {"id": primary_cluster, "label": label_str},
                "cost_burden_ratio":  primary_burden,
                "cost_burden_label":  burden_label,
                "recommendation":     recommendation,
                "age_used":           safe_age,
                "notes":              notes,
            },
            "warning": {"anomaly_flag": anomaly_flag, "messages": anomalies},
            "alternative_interpretation": alternative,
            # Flat (main.js)
            "cluster":           primary_cluster,
            "cost_burden_ratio": primary_burden,
            "cost_burden_label": burden_label,
            "recommendation":    recommendation,
            "anomaly_flag":      anomaly_flag,
            "anomaly_reasons":   anomalies,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=5000)
