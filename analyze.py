"""
Router: /api/v1/analyze
Faktor dominan + rekomendasi AI.
Menggunakan Groq API dengan fallback rule-based.
"""

import os
import json
import re
import traceback

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from groq import Groq

# ── Load ENV ──────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Groq Client ───────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or "llama-3.1-8b-instant"

# ── FastAPI Router ────────────────────────────────────────────
router = APIRouter(
    prefix="/api/v1/analyze",
    tags=["Analyze"],
)


# ─────────────────────────────────────────────────────────────
# Schemas — Input
# ─────────────────────────────────────────────────────────────

class StudentFeatures(BaseModel):
    """
    Raw academic and socio-economic features of a student.
    No identity fields here — identity lives in StudentAnalysisRequest.
    """
    Hours_Studied: int = Field(..., ge=0, le=48)
    Attendance: float = Field(..., ge=0, le=100)
    Parental_Involvement: str
    Access_to_Resources: str
    Sleep_Hours: float = Field(..., ge=0, le=24)
    Previous_Scores: float = Field(..., ge=0, le=100)
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: int = Field(..., ge=0)
    Family_Income: str
    Teacher_Quality: str
    Peer_Influence: str
    Physical_Activity: int = Field(..., ge=0)
    Parental_Education_Level: str


class PredictionContext(BaseModel):
    """
    Result from /predict that needs to be forwarded for analysis.
    """
    risk_category: str
    confidence: float
    predicted_exam_score: float


class StudentAnalysisRequest(BaseModel):
    """
    Full analysis request payload.

    Identity:
      - student_id  : required, opaque identifier (e.g. "STU-001")
      - name        : optional, only surfaced in the response output

    Keeping name optional means callers that don't store names
    (or haven't fetched them yet) can still call this endpoint;
    the response will echo whatever is provided.
    """
    student_id: str = Field(..., example="STU-001")
    # name: Optional[str] = Field(None, example="Airin Sastra")

    features: StudentFeatures
    prediction: PredictionContext


# ─────────────────────────────────────────────────────────────
# Schemas — Output
# ─────────────────────────────────────────────────────────────

class DominantFactor(BaseModel):
    factor: str
    value: str
    status: str   # "good" | "warning" | "danger" | "info"
    note: str


class RecommendationItem(BaseModel):
    title: str
    description: str
    action: str


class StudentMeta(BaseModel):
    """Identity fields echoed back in every response."""
    student_id: str
    # name: Optional[str] = None
    risk_category: str


class DominantFactorsResponse(BaseModel):
    success: bool = True
    student: StudentMeta
    source: str           # "groq" | "rule_based"
    factors: List[DominantFactor]


class RecommendationsResponse(BaseModel):
    success: bool = True
    student: StudentMeta
    source: str           # "groq" | "rule_based"
    recommendations: List[RecommendationItem]


# ─────────────────────────────────────────────────────────────
# Groq Call
# ─────────────────────────────────────────────────────────────

async def _call_llm(prompt: str, max_tokens: int = 512) -> str:

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY tidak di-set.")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )

    print("\n=== GROQ DEBUG ===")
    print("finish_reason:", response.choices[0].finish_reason)
    print("usage:", response.usage)

    content = response.choices[0].message.content

    print("content length:", len(content))
    print(content[:2000])
    print("==================")

    return content.strip()


# ─────────────────────────────────────────────────────────────
# JSON Parser
# ─────────────────────────────────────────────────────────────

def _parse_json(text: str):

    text = re.sub(
        r"```json|```",
        "",
        text
    ).strip()

    start = text.find("[")

    if start == -1:
        raise ValueError(
            f"Tidak ada JSON array: {text[:300]}"
        )

    json_text = text[start:]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON invalid: {e}\n\n{json_text[:500]}"
        )

# ─────────────────────────────────────────────────────────────
# Prompt Builders
# Nama siswa disertakan di prompt agar LLM bisa personalisasi
# narasi, tapi identitas utama tetap student_id di payload.
# ─────────────────────────────────────────────────────────────

def _display_name(req):
    return req.student_id


# ================================================================
# PROMPT v3 — EduPredict AI · GenAI (Groq/Gemma)
# Perubahan dari v2:
# - Note lebih ringkas, on point, tidak bertele-tele (max 25 kata)
# - Bahasa lebih natural dan ramah untuk guru Indonesia
# - Schema recommendations tetap {title, description, action}
#   tapi description dan action lebih padat
# - Guardrail inkonsistensi: status WAJIB selaras risk_category
# ================================================================


def _classify_numerik(f) -> dict:
    """
    Klasifikasikan fitur numerik ke level kondisi.
    Digunakan internal — tidak terekspos ke output.
    Berdasarkan distribusi aktual dataset (6.607 siswa):
      Attendance: min=60, max=100, mean=80
      Hours_Studied: min=4, max=36, mean=20
      Previous_Scores: min=50, max=100, mean=75
      Sleep_Hours: min=4, max=10, ideal=6-8
      Tutoring_Sessions: min=0, max=3.5, mean=1.4
    """
    levels = {}

    att = f.Attendance
    if att >= 90:
        levels['attendance'] = 'optimal'
    elif att >= 80:
        levels['attendance'] = 'baik'
    elif att >= 70:
        levels['attendance'] = 'perlu_perhatian'
    else:
        levels['attendance'] = 'kritis'

    hrs = f.Hours_Studied
    if hrs >= 24:
        levels['hours'] = 'optimal'
    elif hrs >= 16:
        levels['hours'] = 'baik'
    elif hrs >= 10:
        levels['hours'] = 'perlu_perhatian'
    else:
        levels['hours'] = 'kritis'

    ps = f.Previous_Scores
    if ps >= 88:
        levels['prev'] = 'optimal'
    elif ps >= 63:
        levels['prev'] = 'baik'
    elif ps >= 55:
        levels['prev'] = 'perlu_perhatian'
    else:
        levels['prev'] = 'kritis'

    slp = f.Sleep_Hours
    levels['sleep'] = 'optimal' if 6 <= slp <= 8 else 'perlu_perhatian'

    levels['tutoring'] = 'baik' if f.Tutoring_Sessions >= 1 else 'perlu_perhatian'

    return levels


def _status_rule(risk_category: str, level: str) -> str:
    """
    Tentukan status yang konsisten antara risk_category dan kondisi fitur.
    Mencegah inkonsistensi seperti: High Risk + status 'good' di semua faktor.

    Aturan:
    - High Risk  → status buruk minimal 'warning', tidak boleh semua 'good'
    - Medium Risk → campuran 'warning' dan 'good' diperbolehkan
    - Low Risk   → mayoritas 'good', boleh ada 'info'
    """
    if risk_category == "High":
        mapping = {
            'optimal':         'warning',  # tetap apresiasi tapi tetap waspada
            'baik':            'warning',
            'perlu_perhatian': 'danger',
            'kritis':          'danger',
        }
    elif risk_category == "Medium":
        mapping = {
            'optimal':         'good',
            'baik':            'good',
            'perlu_perhatian': 'warning',
            'kritis':          'danger',
        }
    else:  # Low Risk
        mapping = {
            'optimal':         'good',
            'baik':            'good',
            'perlu_perhatian': 'info',
            'kritis':          'warning',
        }
    return mapping.get(level, 'info')


def _factor_prompt(req) -> str:
    f = req.features
    p = req.prediction
    levels = _classify_numerik(f)

    # Tentukan status yang konsisten dengan risk_category
    att_status   = _status_rule(p.risk_category, levels['attendance'])
    hrs_status   = _status_rule(p.risk_category, levels['hours'])
    prev_status  = _status_rule(p.risk_category, levels['prev'])
    motiv_level  = 'optimal' if f.Motivation_Level == 'High' else ('baik' if f.Motivation_Level == 'Medium' else 'kritis')
    motiv_status = _status_rule(p.risk_category, motiv_level)

    # Konteks risiko — memandu AI secara internal
    risk_context = {
        "High":   "Kondisi siswa membutuhkan perhatian serius dan tindakan segera dari guru.",
        "Medium": "Kondisi siswa perlu dipantau agar tidak memburuk.",
        "Low":    "Kondisi siswa baik dan perlu dipertahankan.",
    }.get(p.risk_category, "")

    return f"""
Kamu adalah asisten akademik yang membantu guru memahami kondisi belajar siswanya.
Gunakan bahasa yang hangat, jelas, dan mudah dipahami — seperti rekan guru yang berbagi informasi.

KONDISI SISWA: {risk_context}

DATA SISWA:
Risiko          : {p.risk_category} ({p.confidence:.0f}% keyakinan model)
Kehadiran       : {f.Attendance}%
Jam Belajar     : {f.Hours_Studied} jam/minggu
Jam Tidur       : {f.Sleep_Hours} jam/malam
Nilai Rapor     : {f.Previous_Scores}/100
Motivasi        : {f.Motivation_Level}
Sesi Bimbingan  : {f.Tutoring_Sessions} sesi
Pengaruh Teman  : {f.Peer_Influence}
Keterlibatan Ortu: {f.Parental_Involvement}
Akses Internet  : {f.Internet_Access}
Sumber Belajar  : {f.Access_to_Resources}
Pendapatan Kel. : {f.Family_Income}
Kualitas Guru   : {f.Teacher_Quality}
Aktivitas Fisik : {f.Physical_Activity}x/minggu
Pendidikan Ortu : {f.Parental_Education_Level}

PANDUAN STATUS YANG SUDAH DITENTUKAN (ikuti ini, jangan ubah):
- Kehadiran ({f.Attendance}%)   → status WAJIB: "{att_status}"
- Nilai Rapor ({f.Previous_Scores}/100) → status WAJIB: "{prev_status}"
- Motivasi ({f.Motivation_Level})        → status WAJIB: "{motiv_status}"
- Jam Belajar ({f.Hours_Studied} jam)  → status WAJIB: "{hrs_status}"

TUGAS:
Tulis analisis 4 faktor akademik dominan sesuai data siswa di atas.
Faktor yang WAJIB ada (urutan tetap):
1. Kehadiran
2. Nilai Akademik
3. Motivasi Belajar
4. Jam Belajar

ATURAN PENULISAN "note":
- Maksimal 20 kata — singkat dan langsung ke poin
- Sertakan nilai aktual siswa (angka/level)
- Jelaskan kondisinya secara konkret, bukan umum
- Bahasa natural, hangat, tidak kaku
- JANGAN gunakan kata: dataset, model, sistem, AI, pelatihan

CONTOH note BAGUS (20 kata, natural):
  "Kehadiran 65% cukup mengkhawatirkan — siswa kehilangan hampir sepertiga waktu belajar di kelas."
  "Motivasi yang rendah membuat siswa sulit konsisten mengerjakan tugas dan mengikuti pelajaran."

CONTOH note KURANG BAGUS:
  "Kehadiran rendah dan perlu perhatian." ← terlalu generik
  "Kehadiran siswa sangat rendah, hal ini menunjukkan bahwa siswa tersebut memiliki masalah..." ← terlalu panjang

ATURAN OUTPUT:
- JSON array murni, tepat 4 item, urutan sesuai faktor wajib
- Tidak ada teks di luar array, tidak ada markdown

FORMAT:
[
  {{
    "factor": "Kehadiran",
    "value": "{f.Attendance}%",
    "status": "{att_status}",
    "note": "Tulis di sini — max 20 kata, sertakan angka aktual."
  }},
  {{
    "factor": "Nilai Akademik",
    "value": "{f.Previous_Scores}/100",
    "status": "{prev_status}",
    "note": "Tulis di sini — max 20 kata, sertakan angka aktual."
  }},
  {{
    "factor": "Motivasi Belajar",
    "value": "{f.Motivation_Level}",
    "status": "{motiv_status}",
    "note": "Tulis di sini — max 20 kata, sertakan level aktual."
  }},
  {{
    "factor": "Jam Belajar",
    "value": "{f.Hours_Studied} jam/minggu",
    "status": "{hrs_status}",
    "note": "Tulis di sini — max 20 kata, sertakan angka aktual."
  }}
]
"""


def _recommendation_prompt(req) -> str:
    f = req.features
    p = req.prediction
    levels = _classify_numerik(f)

    risk_tone = {
        "High": (
            "Siswa butuh bantuan segera. "
            "Tulis rekomendasi yang tegas, konkret, dan bisa dimulai minggu ini. "
            "Nada: serius tapi tetap suportif dan tidak menghakimi."
        ),
        "Medium": (
            "Siswa perlu dorongan untuk berkembang. "
            "Tulis rekomendasi yang membangun dan bisa diterapkan bertahap. "
            "Nada: encouragement, optimis, suportif."
        ),
        "Low": (
            "Siswa sudah bagus! "
            "Tulis rekomendasi yang mengapresiasi dan mendorong konsistensi. "
            "Nada: hangat, bangga, positif."
        ),
    }.get(p.risk_category, "")

    # Identifikasi faktor kritis untuk fokus rekomendasi — internal
    critical = []
    if levels['attendance'] in ('kritis', 'perlu_perhatian'):
        critical.append(f"kehadiran {f.Attendance}%")
    if levels['hours'] in ('kritis', 'perlu_perhatian'):
        critical.append(f"jam belajar {f.Hours_Studied} jam/minggu")
    if f.Motivation_Level == "Low":
        critical.append("motivasi rendah")
    if levels['prev'] in ('kritis', 'perlu_perhatian'):
        critical.append(f"nilai rapor {f.Previous_Scores}/100")
    if f.Parental_Involvement == "Low":
        critical.append("keterlibatan orang tua kurang")
    if f.Peer_Influence == "Negative":
        critical.append("pengaruh teman negatif")
    if f.Access_to_Resources == "Low":
        critical.append("sumber belajar terbatas")
    if f.Family_Income == "Low":
        critical.append("kondisi ekonomi keluarga rendah")

    focus = (
        f"Prioritaskan rekomendasi pada: {', '.join(critical)}."
        if critical else
        "Siswa tidak punya faktor kritis — fokus pada penguatan dan apresiasi."
    )

    return f"""
Kamu adalah asisten akademik yang membantu guru merancang langkah nyata untuk membina siswanya.
Gunakan bahasa yang hangat, praktis, dan mudah dipahami guru Indonesia.

ARAHAN UTAMA:
{risk_tone}

DATA SISWA:
Risiko          : {p.risk_category} ({p.confidence:.0f}% keyakinan model)
Kehadiran       : {f.Attendance}%
Jam Belajar     : {f.Hours_Studied} jam/minggu
Jam Tidur       : {f.Sleep_Hours} jam/malam
Nilai Rapor     : {f.Previous_Scores}/100
Motivasi        : {f.Motivation_Level}
Sesi Bimbingan  : {f.Tutoring_Sessions} sesi
Pengaruh Teman  : {f.Peer_Influence}
Keterlibatan Ortu: {f.Parental_Involvement}
Akses Internet  : {f.Internet_Access}
Sumber Belajar  : {f.Access_to_Resources}
Pendapatan Kel. : {f.Family_Income}
Kualitas Guru   : {f.Teacher_Quality}
Aktivitas Fisik : {f.Physical_Activity}x/minggu
Pendidikan Ortu : {f.Parental_Education_Level}

FOKUS (panduan internal, jangan tampilkan ke output):
{focus}

PANDUAN PENULISAN (ikuti ketat):

"title" — 5–8 kata, jelas, aksi nyata
  BAGUS : "Ajak Diskusi Santai tentang Hambatan Belajar"
  KURANG: "Perhatikan Kondisi Siswa Lebih Lanjut"

"description" — 2 kalimat, max 35 kata total
  - Kalimat 1: kenapa ini penting untuk siswa INI (sebutkan angka/kondisi aktualnya)
  - Kalimat 2: dampak jika dilakukan atau tidak dilakukan
  - Nada hangat, tidak menggurui, tidak kaku
  - JANGAN sebut: dataset, model, AI, sistem

"action" — 1 kalimat, max 20 kata, langsung bisa dikerjakan guru
  - Sebutkan caranya atau siapa yang terlibat
  BAGUS : "Hubungi orang tua minggu ini untuk diskusi singkat tentang kebiasaan belajar di rumah."
  KURANG: "Lakukan komunikasi dengan pihak terkait."

ATURAN OUTPUT:
- JSON array murni, tepat 4 item
- Tidak ada teks di luar array, tidak ada markdown

FORMAT:
[
  {{
    "title": "5-8 kata judul aksi konkret",
    "description": "2 kalimat max 35 kata total berbasis kondisi aktual siswa.",
    "action": "1 kalimat langkah yang bisa langsung dilakukan guru."
  }},
  {{
    "title": "...",
    "description": "...",
    "action": "..."
  }},
  {{
    "title": "...",
    "description": "...",
    "action": "..."
  }},
  {{
    "title": "...",
    "description": "...",
    "action": "..."
  }}
]
"""



# ─────────────────────────────────────────────────────────────
# Rule-based Fallback — Dominant Factors
# ─────────────────────────────────────────────────────────────

def _rule_factors(req: StudentAnalysisRequest) -> List[DominantFactor]:

    f = req.features
    factors = []

    # ── 1. KEHADIRAN (WAJIB) ──────────────────────────────────
    if f.Attendance < 75:
        attendance_status = "danger"
        attendance_note = "Kehadiran rendah dan perlu perhatian serius."
    elif f.Attendance < 85:
        attendance_status = "warning"
        attendance_note = "Kehadiran cukup baik tetapi masih perlu ditingkatkan."
    elif f.Attendance < 95:
        attendance_status = "good"
        attendance_note = "Kehadiran siswa sudah baik dan cukup konsisten."
    else:
        attendance_status = "good"
        attendance_note = "Kehadiran sangat baik dan menunjukkan disiplin tinggi."

    factors.append(DominantFactor(
        factor="Kehadiran",
        value=f"{f.Attendance:.0f}%",
        status=attendance_status,
        note=attendance_note,
    ))

    # ── 2. NILAI SEBELUMNYA (WAJIB) ───────────────────────────
    if f.Previous_Scores < 50:
        score_status = "danger"
        score_note = "Nilai akademik sangat rendah dan perlu pendampingan intensif."
    elif f.Previous_Scores < 70:
        score_status = "warning"
        score_note = "Nilai masih berada di bawah target optimal."
    elif f.Previous_Scores < 85:
        score_status = "good"
        score_note = "Nilai akademik cukup baik dan stabil."
    else:
        score_status = "good"
        score_note = "Nilai akademik sangat baik dan konsisten."

    factors.append(DominantFactor(
        factor="Nilai akademik",
        value=f"{f.Previous_Scores:.0f}/100",
        status=score_status,
        note=score_note,
    ))

    # ── 3. MOTIVASI (WAJIB) ───────────────────────────────────
    if f.Motivation_Level == "Low":
        motivation_status = "danger"
        motivation_note = "Motivasi belajar rendah dan perlu dorongan tambahan."
    elif f.Motivation_Level == "Medium":
        motivation_status = "info"
        motivation_note = "Motivasi cukup baik namun masih dapat ditingkatkan."
    else:
        motivation_status = "good"
        motivation_note = "Motivasi belajar sangat baik."

    factors.append(DominantFactor(
        factor="Motivasi belajar",
        value=f.Motivation_Level,
        status=motivation_status,
        note=motivation_note,
    ))

    # ── 4. JAM BELAJAR (WAJIB) ────────────────────────────────
    if f.Hours_Studied < 3:
        study_status = "danger"
        study_note = "Jam belajar sangat kurang."
    elif f.Hours_Studied < 7:
        study_status = "warning"
        study_note = "Jam belajar masih kurang optimal."
    elif f.Hours_Studied < 12:
        study_status = "info"
        study_note = "Jam belajar cukup baik."
    else:
        study_status = "good"
        study_note = "Jam belajar sangat baik dan konsisten."

    factors.append(DominantFactor(
        factor="Jam belajar",
        value=f"{f.Hours_Studied} jam/minggu",
        status=study_status,
        note=study_note,
    ))

    # Sort: danger → warning → info → good
    priority = {"danger": 0, "warning": 1, "info": 2, "good": 3}
    factors.sort(key=lambda x: priority.get(x.status, 9))

    return factors


# ─────────────────────────────────────────────────────────────
# Rule-based Fallback — Recommendations
# ─────────────────────────────────────────────────────────────

def _rule_recommendations(req: StudentAnalysisRequest) -> List[RecommendationItem]:

    f = req.features
    recs = []

    # ── Kehadiran ─────────────────────────────────────────────
    if f.Attendance < 75:
        recs.append(RecommendationItem(
            text="Tingkatkan konsistensi kehadiran siswa karena absensi masih rendah."
        ))
    elif f.Attendance < 90:
        recs.append(RecommendationItem(
            text="Pertahankan kehadiran dan usahakan lebih konsisten setiap minggu."
        ))
    else:
        recs.append(RecommendationItem(
            text="Kehadiran siswa sangat baik dan perlu dipertahankan."
        ))

    # ── Jam Belajar ───────────────────────────────────────────
    if f.Hours_Studied < 5:
        recs.append(RecommendationItem(
            text="Tambahkan jam belajar menjadi minimal 8–10 jam per minggu."
        ))
    elif f.Hours_Studied < 10:
        recs.append(RecommendationItem(
            text="Tingkatkan konsistensi belajar agar hasil akademik lebih optimal."
        ))
    else:
        recs.append(RecommendationItem(
            text="Jam belajar siswa sudah baik dan menunjukkan disiplin belajar yang positif."
        ))

    # ── Motivasi ──────────────────────────────────────────────
    if f.Motivation_Level == "Low":
        recs.append(RecommendationItem(
            text="Berikan motivasi dan target belajar kecil agar siswa lebih percaya diri."
        ))
    elif f.Motivation_Level == "Medium":
        recs.append(RecommendationItem(
            text="Dorong siswa untuk lebih aktif dan konsisten dalam belajar."
        ))
    else:
        recs.append(RecommendationItem(
            text="Motivasi belajar siswa sangat baik dan perlu terus diapresiasi."
        ))

    # ── Faktor ke-4 (Dinamis) ─────────────────────────────────
    if f.Sleep_Hours < 6:
        dynamic = "Perbaiki pola tidur siswa karena waktu istirahat masih kurang."
    elif f.Peer_Influence == "Negative":
        dynamic = "Perhatikan lingkungan pertemanan siswa agar tetap mendukung proses belajar."
    elif f.Parental_Involvement == "Low":
        dynamic = "Ajak orang tua lebih aktif mendampingi proses belajar siswa di rumah."
    elif f.Tutoring_Sessions == 0 and f.Previous_Scores < 70:
        dynamic = "Pertimbangkan bimbingan belajar tambahan untuk membantu pemahaman materi."
    elif f.Physical_Activity < 2:
        dynamic = "Dorong siswa lebih aktif secara fisik untuk membantu fokus dan konsentrasi."
    elif (
        f.Attendance >= 95
        and f.Previous_Scores >= 85
        and f.Motivation_Level == "High"
        and f.Hours_Studied >= 10
    ):
        dynamic = "Performa akademik siswa sangat baik. Pertahankan konsistensi dan semangat belajarnya."
    else:
        dynamic = "Pantau perkembangan akademik siswa secara berkala agar performa tetap stabil."

    recs.append(RecommendationItem(text=dynamic))

    return recs[:4]


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.post(
    "/dominant-factors",
    response_model=DominantFactorsResponse,
)
async def dominant_factors(req: StudentAnalysisRequest):

    source = "rule_based"

    try:
        if GROQ_API_KEY:
            raw = await _call_llm(_factor_prompt(req), max_tokens=400)
            parsed = _parse_json(raw)
            factors = [DominantFactor(**item) for item in parsed]
            source = "groq"
        else:
            factors = _rule_factors(req)

    except Exception as e:
        print("\n=== GROQ ERROR dominant-factors ===")
        print(str(e))
        traceback.print_exc()
        print("===================================\n")

        factors = _rule_factors(req)
        source = "rule_based"

    return DominantFactorsResponse(
        success=True,
        student=StudentMeta(
            student_id=req.student_id,
            # name=req.name,
            risk_category=req.prediction.risk_category,
        ),
        source=source,
        factors=factors,
    )


@router.post(
    "/recommendations",
    response_model=RecommendationsResponse,
)
async def recommendations(req: StudentAnalysisRequest):

    source = "rule_based"

    try:
        if GROQ_API_KEY:
            raw = await _call_llm(_recommendation_prompt(req), max_tokens=1500)
            parsed = _parse_json(raw)
            recs = [RecommendationItem(**item) for item in parsed]
            source = "groq"
        else:
            recs = _rule_recommendations(req)

    except Exception as e:
        print("\n=== GROQ ERROR recommendations ===")
        print(str(e))
        traceback.print_exc()
        print("==================================\n")

        recs = _rule_recommendations(req)
        source = "rule_based"

    return RecommendationsResponse(
        success=True,
        student=StudentMeta(
            student_id=req.student_id,
            # name=req.name,
            risk_category=req.prediction.risk_category,
        ),
        source=source,
        recommendations=recs,
    )


# ─────────────────────────────────────────────────────────────
# Test Endpoint
# ─────────────────────────────────────────────────────────────

@router.get("/groq-test")
async def groq_test():
    try:
        result = await _call_llm("Say hello in Indonesian.")
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
