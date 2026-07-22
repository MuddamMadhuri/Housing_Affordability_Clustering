"""
supabase_seed.py  —  One-time data upload script
=================================================
Run this ONCE on your local machine after creating the Supabase project
and running schema.sql.

Steps:
  1. Create a .env file (copy .env.example) and fill in your keys.
  2. python supabase_seed.py

What it does:
  • Computes cluster_stats from refined_metadata.csv  → uploads to Supabase
  • Samples 2 000 scatter rows                        → uploads to Supabase
  • Uploads kmeans_model.pkl + scaler.pkl             → Supabase Storage (bucket: models)
  • Seeds the default admin user                      → users table
"""

import os, sys, hashlib, io
import pandas as pd

# ── credentials ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SUPABASE_URL         = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("❌  Missing credentials.")
    print("    Create a .env file with:")
    print("      SUPABASE_URL=https://xxxx.supabase.co")
    print("      SUPABASE_SERVICE_KEY=eyJh...  (service_role key)")
    sys.exit(1)

from supabase import create_client
print(f"🔗  Connecting to {SUPABASE_URL} …")
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
print("✅  Connected.\n")

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(BASE_DIR, "refined_metadata.csv")
ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "cluster_assignments.csv")
MODEL_PATH       = os.path.join(BASE_DIR, "kmeans_model.pkl")
SCALER_PATH      = os.path.join(BASE_DIR, "scaler.pkl")

# ── 1. cluster_stats ─────────────────────────────────────────
print("📊  Computing cluster_stats …")
if not os.path.exists(DATA_PATH):
    print(f"⚠️   {DATA_PATH} not found — skipping.\n")
else:
    meta  = pd.read_csv(DATA_PATH)
    asgn  = pd.read_csv(ASSIGNMENTS_PATH)
    if "Cluster_Label" in asgn.columns:
        meta["Cluster_Label"] = asgn["Cluster_Label"]

    cols  = [c for c in ["ZINC2","COSTMED","cost_burden_ratio","AGE1","BEDRMS"] if c in meta.columns]
    stats = meta.groupby("Cluster_Label")[cols].mean()
    stats["Count"] = meta["Cluster_Label"].value_counts()

    rows = [
        {
            "cluster_id":        int(cid),
            "zinc2":             float(r.get("ZINC2", 0)),
            "costmed":           float(r.get("COSTMED", 0)),
            "cost_burden_ratio": float(r.get("cost_burden_ratio", 0)),
            "age1":              float(r.get("AGE1", 0)),
            "bedrms":            float(r.get("BEDRMS", 0)),
            "count":             int(r.get("Count", 0)),
        }
        for cid, r in stats.iterrows()
    ]
    sb.table("cluster_stats").upsert(rows, on_conflict="cluster_id").execute()
    print(f"✅  {len(rows)} cluster_stats rows uploaded.\n")

# ── 2. scatter_data ──────────────────────────────────────────
print("🔵  Uploading scatter_data sample (2 000 rows) …")
if os.path.exists(DATA_PATH):
    sample = meta.sample(n=min(2000, len(meta)), random_state=42).fillna(0)
    scatter = [
        {
            "zinc2":             float(r["ZINC2"]),
            "cost_burden_ratio": float(r["cost_burden_ratio"]),
            "cluster_label":     int(r["Cluster_Label"]),
            "costmed":           float(r["COSTMED"]),
        }
        for _, r in sample.iterrows()
    ]
    # Clear old rows then batch-insert
    sb.table("scatter_data").delete().gt("id", 0).execute()
    for i in range(0, len(scatter), 500):
        sb.table("scatter_data").insert(scatter[i:i+500]).execute()
    print(f"✅  {len(scatter)} scatter rows uploaded.\n")

# ── 3. model files → Supabase Storage ───────────────────────
print("🤖  Uploading model files to Supabase Storage (bucket: models) …")
for fname, fpath in [("kmeans_model.pkl", MODEL_PATH), ("scaler.pkl", SCALER_PATH)]:
    if not os.path.exists(fpath):
        print(f"   ⚠️  {fpath} not found — skipping.")
        continue
    with open(fpath, "rb") as f:
        data = f.read()
    try:
        # Remove existing then upload fresh
        sb.storage.from_("models").remove([fname])
    except Exception:
        pass
    sb.storage.from_("models").upload(
        path=fname,
        file=data,
        file_options={"content-type": "application/octet-stream", "upsert": "true"},
    )
    print(f"   ✅  {fname} ({len(data)//1024} KB) uploaded.")
print()

# ── 4. default admin user ────────────────────────────────────
print("👤  Seeding default admin user …")
admin_hash = hashlib.sha256("housing123".encode()).hexdigest()
sb.table("users").upsert(
    {"username": "admin", "full_name": "Admin User",
     "email": "admin@housing.ai", "password_hash": admin_hash},
    on_conflict="username",
).execute()
print("✅  Admin seeded  (username: admin  |  password: housing123)\n")

print("🎉  Seed complete!")
print("    Set SUPABASE_URL + SUPABASE_SERVICE_KEY on Render, then push & deploy.")
