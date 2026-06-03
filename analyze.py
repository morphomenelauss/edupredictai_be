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
        "High":   "Tegas, konkret, bisa dimulai minggu ini. Nada serius tapi suportif.",
        "Medium": "Membangun, bisa diterapkan bertahap. Nada optimis dan encouragement.",
        "Low":    "Apresiasi dan dorong konsistensi. Nada hangat dan positif.",
    }.get(p.risk_category, "")

    critical = []
    if levels['attendance'] in ('kritis', 'perlu_perhatian'):
        critical.append(f"kehadiran {f.Attendance}%")
    if levels['hours'] in ('kritis', 'perlu_perhatian'):
        critical.append(f"jam belajar {f.Hours_Studied} jam/minggu")
    if f.Motivation_Level == "Low":
        critical.append("motivasi rendah")
    if levels['prev'] in ('kritis', 'perlu_perhatian'):
        critical.append(f"nilai {f.Previous_Scores}/100")
    if f.Parental_Involvement == "Low":
        critical.append("orang tua kurang terlibat")
    if f.Peer_Influence == "Negative":
        critical.append("pengaruh teman negatif")

    focus = f"Fokus: {', '.join(critical)}." if critical else "Fokus: penguatan dan apresiasi."

    return f"""Kamu asisten akademik. Bantu guru dengan rekomendasi praktis dan hangat.

ARAHAN: {risk_tone}
{focus}

DATA: Risiko {p.risk_category} | Kehadiran {f.Attendance}% | Belajar {f.Hours_Studied}j/minggu | Nilai {f.Previous_Scores} | Motivasi {f.Motivation_Level} | Ortu {f.Parental_Involvement} | Teman {f.Peer_Influence}

ATURAN KETAT:
- "title": 5–7 kata, aksi konkret
- "description": 1 kalimat, max 20 kata, sebut angka/kondisi aktual
- "action": 1 kalimat, max 15 kata, langsung bisa dikerjakan guru
- Jangan sebut: dataset, model, AI, sistem

OUTPUT: JSON array murni, tepat 4 item, tanpa teks lain.

[{{"title":"...","description":"...","action":"..."}},{{"title":"...","description":"...","action":"..."}},{{"title":"...","description":"...","action":"..."}},{{"title":"...","description":"...","action":"..."}}]"""



# ─────────────────────────────────────────────────────────────
# Rule-based Fallback — Dominant Factors
# ─────────────────────────────────────────────────────────────

def _rule_factors(req: StudentAnalysisRequest) -> List[DominantFactor]:
    f = req.features
    p = req.prediction
    factors = []

    # ── 1. KEHADIRAN ─────────────────────────────────────────
    att = f.Attendance
    if att < 65:
        att_status = "danger"
        att_note = f"Kehadiran {att:.0f}% sangat kritis — siswa kehilangan lebih dari sepertiga waktu belajar."
    elif att < 75:
        att_status = "danger"
        att_note = f"Kehadiran {att:.0f}% rendah dan berisiko tertinggal banyak materi penting."
    elif att < 80:
        att_status = "warning"
        att_note = f"Kehadiran {att:.0f}% masih di bawah standar — perlu konsistensi lebih tiap minggu."
    elif att < 85:
        att_status = "warning"
        att_note = f"Kehadiran {att:.0f}% cukup, tapi masih ada ruang untuk lebih konsisten hadir."
    elif att < 90:
        att_status = "good"
        att_note = f"Kehadiran {att:.0f}% sudah baik dan menunjukkan kedisiplinan yang cukup stabil."
    elif att < 95:
        att_status = "good"
        att_note = f"Kehadiran {att:.0f}% sangat baik — siswa hampir selalu hadir dan mengikuti pelajaran."
    else:
        att_status = "good"
        att_note = f"Kehadiran {att:.0f}% sempurna — siswa sangat disiplin dan tidak melewatkan kelas."

    # Override status jika High Risk
    if p.risk_category == "High" and att_status == "good":
        att_status = "warning"

    factors.append(DominantFactor(
        factor="Kehadiran",
        value=f"{att:.0f}%",
        status=att_status,
        note=att_note,
    ))

    # ── 2. NILAI AKADEMIK ─────────────────────────────────────
    ps = f.Previous_Scores
    if ps < 50:
        score_status = "danger"
        score_note = f"Nilai {ps:.0f}/100 sangat rendah — siswa perlu pendampingan intensif segera."
    elif ps < 60:
        score_status = "danger"
        score_note = f"Nilai {ps:.0f}/100 masih jauh dari target — pemahaman materi perlu diperkuat."
    elif ps < 70:
        score_status = "warning"
        score_note = f"Nilai {ps:.0f}/100 berada di bawah rata-rata kelas dan perlu ditingkatkan."
    elif ps < 78:
        score_status = "warning"
        score_note = f"Nilai {ps:.0f}/100 cukup namun belum optimal — masih ada potensi yang bisa digali."
    elif ps < 85:
        score_status = "good"
        score_note = f"Nilai {ps:.0f}/100 cukup baik dan menunjukkan pemahaman materi yang memadai."
    elif ps < 92:
        score_status = "good"
        score_note = f"Nilai {ps:.0f}/100 sangat baik — siswa memahami materi dengan konsisten."
    else:
        score_status = "good"
        score_note = f"Nilai {ps:.0f}/100 luar biasa — siswa menguasai materi dengan sangat baik."

    if p.risk_category == "High" and score_status == "good":
        score_status = "warning"

    factors.append(DominantFactor(
        factor="Nilai Akademik",
        value=f"{ps:.0f}/100",
        status=score_status,
        note=score_note,
    ))

    # ── 3. MOTIVASI ──────────────────────────────────────────
    motiv = f.Motivation_Level
    hours = f.Hours_Studied

    if motiv == "Low" and hours < 8:
        motiv_status = "danger"
        motiv_note = f"Motivasi rendah diperparah jam belajar hanya {hours} jam — siswa butuh dorongan segera."
    elif motiv == "Low" and hours >= 8:
        motiv_status = "danger"
        motiv_note = f"Motivasi rendah meski belajar {hours} jam/minggu — kualitas belajar perlu diperhatikan."
    elif motiv == "Medium" and p.risk_category == "High":
        motiv_status = "warning"
        motiv_note = f"Motivasi sedang belum cukup untuk mengejar ketertinggalan — perlu stimulus lebih."
    elif motiv == "Medium" and ps < 70:
        motiv_status = "warning"
        motiv_note = f"Motivasi sedang dengan nilai {ps:.0f} — siswa perlu didorong agar lebih giat belajar."
    elif motiv == "Medium":
        motiv_status = "info"
        motiv_note = f"Motivasi sedang dan masih bisa ditingkatkan agar hasil belajar lebih maksimal."
    elif motiv == "High" and p.risk_category == "High":
        motiv_status = "warning"
        motiv_note = f"Motivasi tinggi tapi belum terefleksi pada performa — arah belajar perlu diperjelas."
    elif motiv == "High" and ps >= 85:
        motiv_status = "good"
        motiv_note = f"Motivasi tinggi sejalan dengan nilai {ps:.0f} — kombinasi yang sangat positif."
    else:
        motiv_status = "good"
        motiv_note = f"Motivasi belajar tinggi — siswa menunjukkan semangat yang perlu terus dijaga."

    factors.append(DominantFactor(
        factor="Motivasi Belajar",
        value=motiv,
        status=motiv_status,
        note=motiv_note,
    ))

    # ── 4. JAM BELAJAR ───────────────────────────────────────
    hrs = f.Hours_Studied
    if hrs < 5:
        hrs_status = "danger"
        hrs_note = f"Jam belajar {hrs} jam/minggu sangat kurang — siswa perlu jadwal belajar yang terstruktur."
    elif hrs < 8:
        hrs_status = "danger"
        hrs_note = f"Hanya {hrs} jam/minggu jauh dari ideal — sulit mengejar materi dengan waktu ini."
    elif hrs < 12:
        hrs_status = "warning"
        hrs_note = f"Jam belajar {hrs} jam/minggu masih di bawah optimal — perlu ditambah secara bertahap."
    elif hrs < 16:
        hrs_status = "warning"
        hrs_note = f"Jam belajar {hrs} jam/minggu cukup, tapi masih ada ruang untuk lebih konsisten."
    elif hrs < 22:
        hrs_status = "good"
        hrs_note = f"Jam belajar {hrs} jam/minggu sudah baik dan mendukung pemahaman materi dengan solid."
    elif hrs < 28:
        hrs_status = "good"
        hrs_note = f"Jam belajar {hrs} jam/minggu sangat baik — siswa menunjukkan dedikasi belajar yang tinggi."
    else:
        hrs_status = "good"
        hrs_note = f"Jam belajar {hrs} jam/minggu sangat intensif — pastikan kualitas dan istirahatnya tetap terjaga."

    if p.risk_category == "High" and hrs_status == "good":
        hrs_status = "warning"

    factors.append(DominantFactor(
        factor="Jam Belajar",
        value=f"{hrs} jam/minggu",
        status=hrs_status,
        note=hrs_note,
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
    p = req.prediction
    recs = []

    # ── REC 1: Kehadiran ─────────────────────────────────────
    att = f.Attendance
    if att < 65:
        recs.append(RecommendationItem(
            title="Selidiki Penyebab Ketidakhadiran Segera",
            description=f"Kehadiran {att:.0f}% sangat kritis — siswa kehilangan lebih dari sepertiga pelajaran. Tanpa intervensi cepat, ketertinggalan materi akan semakin sulit dikejar.",
            action="Hubungi orang tua minggu ini untuk mencari tahu penyebab dan buat kesepakatan perbaikan kehadiran."
        ))
    elif att < 80:
        recs.append(RecommendationItem(
            title="Bangun Kebiasaan Hadir Lebih Konsisten",
            description=f"Kehadiran {att:.0f}% masih di bawah standar dan berpotensi mengganggu pemahaman materi. Pola absensi yang berulang perlu segera diidentifikasi penyebabnya.",
            action="Pantau kehadiran mingguan dan diskusikan hambatannya langsung dengan siswa secara personal."
        ))
    elif att < 90:
        recs.append(RecommendationItem(
            title="Pertahankan dan Tingkatkan Konsistensi Hadir",
            description=f"Kehadiran {att:.0f}% sudah cukup baik namun masih ada beberapa pertemuan yang terlewat. Konsistensi hadir akan memperkuat pemahaman materi secara menyeluruh.",
            action="Berikan apresiasi kecil saat siswa hadir penuh dalam satu minggu untuk membangun motivasi."
        ))
    else:
        recs.append(RecommendationItem(
            title="Apresiasi Kedisiplinan Kehadiran Siswa",
            description=f"Kehadiran {att:.0f}% sangat baik dan mencerminkan kedisiplinan yang perlu dijaga. Siswa yang konsisten hadir cenderung memiliki pemahaman materi yang lebih solid.",
            action="Sampaikan apresiasi langsung kepada siswa agar motivasi dan kedisiplinannya tetap terjaga."
        ))

    # ── REC 2: Jam Belajar + Nilai ───────────────────────────
    hrs = f.Hours_Studied
    ps = f.Previous_Scores
    if hrs < 8 and ps < 70:
        recs.append(RecommendationItem(
            title="Buat Jadwal Belajar Terstruktur Bersama Siswa",
            description=f"Kombinasi {hrs} jam belajar/minggu dan nilai {ps:.0f} adalah sinyal yang perlu ditangani segera. Tanpa jadwal yang jelas, waktu belajar akan terus tidak teroptimalkan.",
            action="Bantu siswa menyusun jadwal belajar harian sederhana yang realistis dan bisa dijalankan konsisten."
        ))
    elif hrs < 12:
        recs.append(RecommendationItem(
            title="Dorong Penambahan Waktu Belajar Bertahap",
            description=f"Dengan {hrs} jam/minggu, siswa belum mencapai durasi belajar yang optimal untuk hasil maksimal. Menambah 2–3 jam per minggu secara bertahap bisa berdampak signifikan pada nilai.",
            action="Sarankan siswa menambah satu sesi belajar 30 menit setiap hari untuk membangun kebiasaan."
        ))
    elif hrs >= 22 and ps >= 85:
        recs.append(RecommendationItem(
            title="Jaga Keseimbangan Belajar dan Istirahat",
            description=f"Jam belajar {hrs} jam/minggu sangat tinggi dan sejalan dengan nilai {ps:.0f} yang memuaskan. Namun intensitas tinggi perlu diimbangi istirahat cukup agar tidak kelelahan.",
            action="Ingatkan siswa untuk menjaga pola tidur dan waktu bermain agar stamina belajar tetap optimal."
        ))
    else:
        recs.append(RecommendationItem(
            title="Tingkatkan Kualitas Sesi Belajar Siswa",
            description=f"Jam belajar {hrs} jam/minggu sudah cukup — fokus selanjutnya adalah efektivitas belajarnya. Belajar dengan teknik yang tepat bisa meningkatkan hasil tanpa menambah durasi.",
            action="Bagikan teknik belajar aktif seperti rangkuman atau latihan soal agar sesi belajar lebih produktif."
        ))

    # ── REC 3: Motivasi + Dukungan ───────────────────────────
    motiv = f.Motivation_Level
    parental = f.Parental_Involvement
    peer = f.Peer_Influence

    if motiv == "Low" and parental == "Low":
        recs.append(RecommendationItem(
            title="Libatkan Orang Tua untuk Bangkitkan Motivasi",
            description=f"Motivasi rendah ditambah kurangnya keterlibatan orang tua menciptakan kondisi belajar yang tidak kondusif. Dukungan dari rumah sangat krusial untuk membangun kepercayaan diri siswa.",
            action="Jadwalkan pertemuan dengan orang tua untuk membahas cara mendukung semangat belajar di rumah."
        ))
    elif motiv == "Low" and peer == "Negative":
        recs.append(RecommendationItem(
            title="Tangani Pengaruh Lingkungan yang Negatif",
            description=f"Motivasi rendah diperparah pengaruh teman yang negatif — dua faktor ini saling memperlemah semangat belajar. Siswa perlu diarahkan ke lingkungan pertemanan yang lebih suportif.",
            action="Ajak siswa bicara personal tentang lingkaran pertemanannya dan dorong bergabung ke kelompok belajar positif."
        ))
    elif motiv == "Low":
        recs.append(RecommendationItem(
            title="Bangun Kepercayaan Diri dengan Target Kecil",
            description=f"Motivasi rendah sering berasal dari rasa tidak mampu yang menumpuk dari waktu ke waktu. Keberhasilan kecil yang konsisten bisa memulihkan semangat belajar siswa secara perlahan.",
            action="Berikan tugas kecil yang bisa diselesaikan siswa dan rayakan keberhasilannya di depan kelas."
        ))
    elif motiv == "Medium" and p.risk_category in ("High", "Medium"):
        recs.append(RecommendationItem(
            title="Perkuat Motivasi agar Tidak Stagnan",
            description=f"Motivasi sedang belum cukup untuk mendorong perubahan signifikan pada kondisi belajar saat ini. Sedikit dorongan yang tepat bisa menggeser motivasi ke level yang lebih tinggi.",
            action="Ceritakan kisah sukses siswa lain yang pernah berada di posisi serupa untuk memicu semangat."
        ))
    else:
        recs.append(RecommendationItem(
            title="Apresiasi Semangat Belajar yang Positif",
            description=f"Motivasi {motiv.lower()} siswa adalah aset berharga yang perlu terus dipupuk oleh guru. Apresiasi yang konsisten akan menjaga semangat ini tetap menyala dalam jangka panjang.",
            action="Berikan pengakuan verbal atau catatan positif di buku siswa untuk menguatkan motivasi belajarnya."
        ))

    # ── REC 4: Faktor Dinamis ────────────────────────────────
    slp = f.Sleep_Hours
    tutoring = f.Tutoring_Sessions
    physical = f.Physical_Activity
    resources = f.Access_to_Resources
    income = f.Family_Income

    if slp < 6:
        recs.append(RecommendationItem(
            title="Perbaiki Pola Tidur untuk Fokus Belajar",
            description=f"Tidur {slp:.0f} jam/malam jauh dari ideal — kurang tidur langsung menurunkan konsentrasi dan daya serap materi. Perbaikan pola tidur bisa meningkatkan performa tanpa menambah jam belajar.",
            action="Diskusikan dengan siswa dan orang tua tentang pentingnya tidur 7–8 jam untuk mendukung belajar."
        ))
    elif tutoring == 0 and ps < 70:
        recs.append(RecommendationItem(
            title="Pertimbangkan Bimbingan Belajar Tambahan",
            description=f"Nilai {ps:.0f}/100 tanpa sesi bimbingan sama sekali menunjukkan siswa perlu dukungan belajar lebih. Bimbingan tambahan bisa membantu mengisi celah pemahaman yang tertinggal.",
            action="Rekomendasikan program remedial sekolah atau bimbingan teman sebaya untuk mata pelajaran terlemah."
        ))
    elif peer == "Negative" and motiv != "Low":
        recs.append(RecommendationItem(
            title="Arahkan ke Lingkungan Pertemanan yang Positif",
            description=f"Pengaruh teman yang negatif bisa perlahan menggerus semangat belajar meskipun motivasi siswa saat ini masih baik. Intervensi dini lebih mudah dilakukan sebelum dampaknya terasa.",
            action="Dorong siswa bergabung dengan kelompok belajar atau ekstrakurikuler yang lingkungannya suportif."
        ))
    elif parental == "Low" and income == "Low":
        recs.append(RecommendationItem(
            title="Berikan Dukungan Ekstra dari Sekolah",
            description=f"Keterbatasan ekonomi dan kurangnya perhatian orang tua membuat siswa lebih bergantung pada dukungan guru. Sekolah bisa mengisi peran penting dalam menjaga motivasi dan akses belajarnya.",
            action="Koordinasikan dengan BK untuk memastikan siswa mendapat akses ke fasilitas belajar yang tersedia di sekolah."
        ))
    elif resources == "Low" and income == "Low":
        recs.append(RecommendationItem(
            title="Pastikan Akses Belajar yang Memadai",
            description=f"Sumber belajar terbatas dengan kondisi ekonomi rendah bisa menjadi hambatan tersembunyi yang signifikan. Memastikan akses ke buku dan internet adalah fondasi penting untuk belajar efektif.",
            action="Hubungkan siswa dengan program bantuan sekolah atau perpustakaan untuk memenuhi kebutuhan belajarnya."
        ))
    elif physical < 2:
        recs.append(RecommendationItem(
            title="Dorong Aktivitas Fisik untuk Konsentrasi",
            description=f"Aktivitas fisik hanya {physical}x/minggu terlalu sedikit — olahraga ringan terbukti meningkatkan fokus dan suasana hati saat belajar. Pergerakan tubuh membantu otak lebih siap menyerap materi.",
            action="Sarankan siswa berjalan kaki atau olahraga ringan minimal 20 menit setiap hari sebelum belajar."
        ))
    elif att >= 90 and ps >= 85 and motiv == "High" and hrs >= 16:
        recs.append(RecommendationItem(
            title="Tantang Siswa dengan Materi yang Lebih Dalam",
            description=f"Performa sangat baik di semua aspek menunjukkan siswa ini siap untuk tantangan yang lebih besar. Memberikan materi pengayaan akan menjaga motivasinya tetap tinggi dan mencegah kebosanan.",
            action="Berikan soal pengayaan atau proyek mandiri yang menantang untuk mengembangkan potensi optimalnya."
        ))
    else:
        recs.append(RecommendationItem(
            title="Pantau Perkembangan Secara Berkala",
            description=f"Kondisi belajar siswa secara keseluruhan perlu dipantau agar tidak ada faktor yang memburuk tanpa terdeteksi. Pemantauan rutin membantu guru bertindak sebelum masalah menjadi lebih besar.",
            action="Lakukan check-in singkat dengan siswa setiap dua minggu untuk memantau perkembangan dan hambatannya."
        ))

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
            raw = await _call_llm(_recommendation_prompt(req), max_tokens=500)
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
