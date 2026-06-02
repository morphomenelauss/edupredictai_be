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
    text: str


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

    print("\n=== GROQ REQUEST START ===")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content

    print("[GROQ SUCCESS]")
    print(content[:1000])

    return content.strip()


# ─────────────────────────────────────────────────────────────
# JSON Parser
# ─────────────────────────────────────────────────────────────

def _parse_json(text: str) -> list:

    text = re.sub(r"```json|```", "", text).strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)

    if match:
        return json.loads(match.group())

    raise ValueError(
        f"Tidak ada JSON array dalam response: {text[:300]}"
    )


# ─────────────────────────────────────────────────────────────
# Prompt Builders
# Nama siswa disertakan di prompt agar LLM bisa personalisasi
# narasi, tapi identitas utama tetap student_id di payload.
# ─────────────────────────────────────────────────────────────

def _display_name(req: StudentAnalysisRequest) -> str:
    """Prefer name if provided, fall back to student_id."""
    return req.name if req.name else req.student_id


def _factor_prompt(req: StudentAnalysisRequest) -> str:

    f = req.features
    p = req.prediction

    return f"""
Kamu adalah AI academic analyst.

Tugas:
Analisis kondisi akademik siswa berdasarkan data yang diberikan.

Pilih TEPAT 4 faktor dominan yang paling mempengaruhi
performa akademik siswa.

4 faktor yang WAJIB ada:
1. Kehadiran
2. Nilai akademik
3. Motivasi belajar
4. Jam belajar

Untuk setiap faktor:
- tentukan status
- jelaskan singkat penyebab atau dampaknya

Data siswa:
Nama: {_display_name(req)}
Kategori Risiko: {p.risk_category}
Confidence Risiko: {p.confidence:.0f}%
Prediksi Nilai: {p.predicted_exam_score:.1f}/100

Kehadiran: {f.Attendance}%
Jam Belajar: {f.Hours_Studied} jam/minggu
Jam Tidur: {f.Sleep_Hours} jam/malam

Nilai Sebelumnya: {f.Previous_Scores}/100
Motivasi Belajar: {f.Motivation_Level}
Sesi Bimbel: {f.Tutoring_Sessions}

Pengaruh Teman: {f.Peer_Influence}
Keterlibatan Orang Tua: {f.Parental_Involvement}

Akses Internet: {f.Internet_Access}
Akses Resource Belajar: {f.Access_to_Resources}

Pendapatan Keluarga: {f.Family_Income}
Kualitas Guru: {f.Teacher_Quality}

Aktivitas Fisik: {f.Physical_Activity}x/minggu
Pendidikan Orang Tua: {f.Parental_Education_Level}

ATURAN OUTPUT:
- Jawab HANYA JSON array
- HARUS tepat 4 item
- Jangan gunakan markdown
- Jangan gunakan ```json
- Jangan beri penjelasan tambahan
- Status hanya boleh:
  "good"
  "warning"
  "danger"
  "info"

Gunakan value yang realistis berdasarkan data siswa.

Contoh format:
[
  {{
    "factor": "Kehadiran",
    "value": "82%",
    "status": "warning",
    "note": "Kehadiran masih kurang konsisten dan mempengaruhi proses belajar."
  }},
  {{
    "factor": "Nilai akademik",
    "value": "68/100",
    "status": "warning",
    "note": "Nilai masih berada di bawah target optimal."
  }},
  {{
    "factor": "Motivasi belajar",
    "value": "Low",
    "status": "danger",
    "note": "Motivasi rendah membuat siswa kurang konsisten belajar."
  }},
  {{
    "factor": "Jam belajar",
    "value": "4 jam/minggu",
    "status": "danger",
    "note": "Jam belajar sangat kurang untuk mencapai hasil maksimal."
  }}
]
"""


def _recommendation_prompt(req: StudentAnalysisRequest) -> str:

    f = req.features
    p = req.prediction

    return f"""
Kamu adalah AI assistant untuk guru sekolah.

Tugas:
Berikan TEPAT 4 rekomendasi konkret, realistis,
dan spesifik berdasarkan kondisi akademik siswa.

Rekomendasi harus:
- praktis
- mudah diterapkan
- relevan dengan kondisi siswa
- fokus meningkatkan performa akademik

Jika performa siswa sudah baik,
berikan apresiasi dan saran untuk mempertahankan performa tersebut.

Data siswa:
Nama: {_display_name(req)}
Kategori Risiko: {p.risk_category}
Prediksi Nilai: {p.predicted_exam_score:.1f}/100

Kehadiran: {f.Attendance}%
Jam Belajar: {f.Hours_Studied} jam/minggu
Motivasi Belajar: {f.Motivation_Level}

Nilai Sebelumnya: {f.Previous_Scores}/100
Sesi Bimbel: {f.Tutoring_Sessions}

Pengaruh Teman: {f.Peer_Influence}
Keterlibatan Orang Tua: {f.Parental_Involvement}

Jam Tidur: {f.Sleep_Hours} jam/malam
Aktivitas Fisik: {f.Physical_Activity}x/minggu

ATURAN OUTPUT:
- Jawab HANYA JSON array
- HARUS tepat 4 item
- Jangan gunakan markdown
- Jangan gunakan ```json
- Jangan beri penjelasan tambahan

Contoh format:
[
  {{
    "text": "Tingkatkan jam belajar menjadi minimal 8-10 jam per minggu."
  }},
  {{
    "text": "Ajak orang tua lebih aktif memantau jadwal belajar siswa di rumah."
  }},
  {{
    "text": "Dorong siswa menjaga konsistensi kehadiran di sekolah."
  }},
  {{
    "text": "Berikan apresiasi atas perkembangan akademik siswa agar motivasi tetap tinggi."
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
            raw = await _call_llm(_recommendation_prompt(req), max_tokens=350)
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
