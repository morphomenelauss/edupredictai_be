# ─────────────────────────────────────────────────────────────
# Tambahkan 2 baris ini di main.py Anda
# ─────────────────────────────────────────────────────────────

# 1. Import router (letakkan di atas, setelah import FastAPI)
from analyze import router as analyze_router

# 2. Daftarkan router (letakkan setelah app = FastAPI(...))
app.include_router(analyze_router)

# ─────────────────────────────────────────────────────────────
# Endpoint yang tersedia setelah ini:
#
#  POST /analyze/dominant-factors   → Faktor Dominan
#  POST /analyze/recommendations    → Rekomendasi AI
#
# Environment variable yang diperlukan (opsional):
#  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
#
# Jika HF_TOKEN tidak di-set, otomatis pakai rule-based fallback.
# ─────────────────────────────────────────────────────────────
