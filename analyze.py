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
# Rule-based — Dominant Factors (dioptimasi: lebih ringkas)
# ─────────────────────────────────────────────────────────────

def _rule_factors(req: StudentAnalysisRequest) -> List[DominantFactor]:
    # Insight harus menggunakan nilai asli (raw) dari user.
    # Jika features_raw tidak tersedia, fallback ke features.
    f = getattr(req, 'features_raw', req.features)
    p = req.prediction
    hr = p.risk_category == "High"
    factors = []

    # ── 1. KEHADIRAN ──────────────────────────────────────────
    att = f.Attendance
    if att < 70:
        s, note = "danger", f"Kehadiran {att:.0f}% sangat rendah — siswa kehilangan banyak materi kelas."
    elif att < 80:
        s, note = "danger" if hr else "warning", f"Kehadiran {att:.0f}% di bawah standar — risiko tertinggal materi cukup tinggi."
    elif att < 90:
        s, note = "warning" if hr else "good", f"Kehadiran {att:.0f}% cukup, masih ada beberapa pertemuan yang terlewat."
    else:
        s, note = "warning" if hr else "good", f"Kehadiran {att:.0f}% sangat baik — siswa konsisten mengikuti pelajaran."
    factors.append(DominantFactor(factor="Kehadiran", value=f"{att:.0f}%", status=s, note=note))

    # ── 2. NILAI AKADEMIK ─────────────────────────────────────
    ps = f.Previous_Scores
    if ps < 60:
        s, note = "danger", f"Nilai {ps:.0f}/100 jauh di bawah target — pemahaman materi perlu diperkuat segera."
    elif ps < 70:
        s, note = "danger" if hr else "warning", f"Nilai {ps:.0f}/100 masih di bawah rata-rata — ada celah pemahaman yang perlu diisi."
    elif ps < 82:
        s, note = "warning" if hr else "good", f"Nilai {ps:.0f}/100 cukup, masih ada potensi yang belum tergali."
    else:
        s, note = "warning" if hr else "good", f"Nilai {ps:.0f}/100 sangat baik — siswa menguasai materi dengan konsisten."
    factors.append(DominantFactor(factor="Nilai Akademik", value=f"{ps:.0f}/100", status=s, note=note))

    # ── 3. MOTIVASI ───────────────────────────────────────────
    motiv = f.Motivation_Level
    hrs   = f.Hours_Studied
    if motiv == "Low":
        s = "danger"
        note = f"Motivasi rendah diperparah jam belajar {hrs} jam/minggu — siswa butuh dorongan segera." if hrs < 10 \
               else f"Motivasi rendah membuat belajar {hrs} jam/minggu kurang efektif — kualitas perlu dijaga."
    elif motiv == "Medium":
        s = "warning" if hr or ps < 70 else "info"
        note = f"Motivasi sedang belum cukup untuk mengejar ketertinggalan saat ini." if hr \
               else f"Motivasi sedang — masih bisa ditingkatkan agar hasil belajar lebih maksimal."
    else:  # High
        s = "warning" if hr else "good"
        note = f"Motivasi tinggi tapi belum terefleksi pada performa — arah belajar perlu diperjelas." if hr \
               else f"Motivasi tinggi sejalan dengan performa — kombinasi yang sangat positif."
    factors.append(DominantFactor(factor="Motivasi Belajar", value=motiv, status=s, note=note))

    # ── 4. JAM BELAJAR ────────────────────────────────────────
    if hrs < 8:
        s, note = "danger", f"Hanya {hrs} jam/minggu — sangat kurang untuk menguasai materi secara memadai."
    elif hrs < 14:
        s, note = "danger" if hr else "warning", f"Jam belajar {hrs} jam/minggu masih di bawah optimal — perlu ditambah bertahap."
    elif hrs < 22:
        s, note = "warning" if hr else "good", f"Jam belajar {hrs} jam/minggu sudah baik dan mendukung pemahaman materi."
    else:
        s, note = "warning" if hr else "good", f"Jam belajar {hrs} jam/minggu sangat intensif — pastikan kualitas dan istirahatnya terjaga."
    factors.append(DominantFactor(factor="Jam Belajar", value=f"{hrs} jam/minggu", status=s, note=note))

    # Sort: danger → warning → info → good
    priority = {"danger": 0, "warning": 1, "info": 2, "good": 3}
    factors.sort(key=lambda x: priority.get(x.status, 9))
    return factors


# ─────────────────────────────────────────────────────────────
# Rule-based — Recommendations (dioptimasi: padat + actionable)
# ─────────────────────────────────────────────────────────────

def _rule_recommendations(req: StudentAnalysisRequest) -> List[RecommendationItem]:
    # Insight harus menggunakan nilai asli (raw) dari user.
    # Jika features_raw tidak tersedia, fallback ke features.
    f = getattr(req, 'features_raw', req.features)
    p = req.prediction
    hr = p.risk_category == "High"
    mr = p.risk_category == "Medium"
    recs = []

    att     = f.Attendance
    ps      = f.Previous_Scores
    hrs     = f.Hours_Studied
    motiv   = f.Motivation_Level
    slp     = f.Sleep_Hours
    tutoring= f.Tutoring_Sessions
    peer    = f.Peer_Influence
    parental= f.Parental_Involvement
    income  = f.Family_Income
    physical= f.Physical_Activity
    resources = f.Access_to_Resources

    # ── REC 1: Kehadiran ──────────────────────────────────────
    if att < 70:
        recs.append(RecommendationItem(
            title="Selidiki Penyebab Ketidakhadiran Segera",
            description=f"Kehadiran {att:.0f}% sangat kritis dan terus menumpuk ketertinggalan materi.",
            action="Hubungi orang tua minggu ini — identifikasi hambatan dan buat komitmen perbaikan bersama."
        ))
    elif att < 80:
        recs.append(RecommendationItem(
            title="Tingkatkan Konsistensi Kehadiran",
            description=f"Kehadiran {att:.0f}% masih di bawah standar dan berisiko mengganggu pemahaman materi.",
            action="Diskusikan hambatan kehadiran langsung dengan siswa dan buat target mingguan yang realistis."
        ))
    elif att < 92:
        recs.append(RecommendationItem(
            title="Jaga Konsistensi Kehadiran yang Sudah Baik",
            description=f"Kehadiran {att:.0f}% sudah cukup baik — tinggal ditingkatkan agar tidak ada materi yang terlewat.",
            action="Beri apresiasi saat siswa hadir penuh satu minggu penuh untuk membangun kebiasaan positif."
        ))
    else:
        recs.append(RecommendationItem(
            title="Apresiasi Kedisiplinan Kehadiran Siswa",
            description=f"Kehadiran {att:.0f}% mencerminkan kedisiplinan tinggi yang mendukung proses belajar.",
            action="Sampaikan apresiasi langsung agar motivasi dan kedisiplinannya terus terjaga."
        ))

    # ── REC 2: Jam Belajar + Nilai ────────────────────────────
    if hrs < 8 and ps < 70:
        recs.append(RecommendationItem(
            title="Buat Jadwal Belajar Harian Bersama",
            description=f"Hanya {hrs} jam/minggu dengan nilai {ps:.0f} adalah kombinasi yang perlu ditangani segera.",
            action="Bantu siswa menyusun jadwal belajar harian 30–45 menit yang sederhana dan bisa langsung dijalankan."
        ))
    elif hrs < 14:
        recs.append(RecommendationItem(
            title="Dorong Penambahan Waktu Belajar Bertahap",
            description=f"Jam belajar {hrs} jam/minggu belum optimal untuk mencapai hasil yang diharapkan.",
            action="Sarankan siswa menambah satu sesi belajar 30 menit setiap hari dimulai dari besok."
        ))
    elif hrs >= 22 and ps >= 85:
        recs.append(RecommendationItem(
            title="Jaga Keseimbangan Belajar dan Istirahat",
            description=f"Jam belajar {hrs} jam/minggu sangat tinggi — perlu diimbangi istirahat agar tidak kelelahan.",
            action="Ingatkan siswa menjaga tidur 7–8 jam dan sisipkan waktu santai agar stamina belajar tetap prima."
        ))
    else:
        recs.append(RecommendationItem(
            title="Tingkatkan Kualitas Sesi Belajar",
            description=f"Jam belajar {hrs} jam/minggu sudah cukup — fokus selanjutnya adalah efektivitasnya.",
            action="Bagikan teknik belajar aktif seperti latihan soal atau membuat rangkuman agar setiap sesi lebih produktif."
        ))

    # ── REC 3: Motivasi + Dukungan ────────────────────────────
    if motiv == "Low" and parental == "Low":
        recs.append(RecommendationItem(
            title="Libatkan Orang Tua untuk Dukung Motivasi",
            description=f"Motivasi rendah tanpa dukungan orang tua membuat siswa kekurangan stimulus dari dua arah.",
            action="Jadwalkan pertemuan singkat dengan orang tua untuk membahas cara mendorong semangat belajar di rumah."
        ))
    elif motiv == "Low" and peer == "Negative":
        recs.append(RecommendationItem(
            title="Arahkan dari Pengaruh Teman yang Negatif",
            description=f"Motivasi rendah diperparah lingkungan pertemanan negatif — dua faktor ini saling melemahkan.",
            action="Ajak siswa bicara personal dan dorong bergabung dengan kelompok belajar yang lebih suportif."
        ))
    elif motiv == "Low":
        recs.append(RecommendationItem(
            title="Bangkitkan Kepercayaan Diri dengan Target Kecil",
            description=f"Motivasi rendah sering berasal dari rasa tidak mampu yang menumpuk secara perlahan.",
            action="Berikan tugas kecil yang bisa diselesaikan siswa hari ini, lalu rayakan keberhasilannya di kelas."
        ))
    elif (hr or mr) and motiv == "Medium":
        recs.append(RecommendationItem(
            title="Tingkatkan Motivasi agar Tidak Stagnan",
            description=f"Motivasi sedang belum cukup untuk mendorong perubahan nyata pada kondisi saat ini.",
            action="Ceritakan kisah sukses siswa lain yang pernah ada di posisi serupa untuk memicu semangat."
        ))
    else:
        recs.append(RecommendationItem(
            title="Pertahankan Semangat Belajar yang Positif",
            description=f"Motivasi {motiv.lower()} siswa adalah aset yang perlu terus dijaga oleh guru.",
            action="Berikan catatan apresiasi singkat di buku siswa secara rutin untuk menguatkan semangatnya."
        ))

    # ── REC 4: Faktor Pendukung Dinamis ───────────────────────
    if slp < 6:
        recs.append(RecommendationItem(
            title="Perbaiki Pola Tidur untuk Fokus Belajar",
            description=f"Tidur {slp:.0f} jam/malam terlalu sedikit — langsung menurunkan konsentrasi dan daya serap materi.",
            action="Minta orang tua memastikan siswa tidur pukul 21.00–22.00 dan bangun teratur setiap hari."
        ))
    elif tutoring == 0 and ps < 70:
        recs.append(RecommendationItem(
            title="Rekomendasikan Bimbingan Belajar Tambahan",
            description=f"Nilai {ps:.0f}/100 tanpa sesi bimbingan menunjukkan ada celah pemahaman yang belum terisi.",
            action="Daftarkan siswa ke program remedial sekolah atau bimbingan teman sebaya untuk mapel terlemahnya."
        ))
    elif peer == "Negative":
        recs.append(RecommendationItem(
            title="Cegah Dampak Lingkungan Pertemanan Negatif",
            description=f"Pengaruh teman negatif bisa perlahan mengikis performa meskipun faktor lain sudah baik.",
            action="Dorong siswa aktif di kelompok belajar atau ekstrakurikuler positif minimal satu kali per minggu."
        ))
    elif parental == "Low" and income == "Low":
        recs.append(RecommendationItem(
            title="Perkuat Dukungan Belajar dari Sekolah",
            description=f"Keterbatasan ekonomi dan minimnya perhatian orang tua membuat siswa lebih bergantung pada guru.",
            action="Koordinasikan dengan BK agar siswa mendapat akses prioritas ke fasilitas dan program bantuan sekolah."
        ))
    elif resources == "Low":
        recs.append(RecommendationItem(
            title="Pastikan Akses Sumber Belajar Terpenuhi",
            description=f"Sumber belajar terbatas bisa jadi hambatan tersembunyi meski faktor lain sudah cukup baik.",
            action="Hubungkan siswa dengan perpustakaan sekolah atau sumber belajar online gratis yang bisa diakses segera."
        ))
    elif physical < 2:
        recs.append(RecommendationItem(
            title="Dorong Aktivitas Fisik Ringan Rutin",
            description=f"Aktivitas fisik hanya {physical}x/minggu — olahraga ringan terbukti meningkatkan fokus belajar.",
            action="Sarankan siswa berjalan kaki atau peregangan 15 menit setiap pagi sebelum memulai aktivitas belajar."
        ))
    elif att >= 90 and ps >= 85 and motiv == "High":
        recs.append(RecommendationItem(
            title="Tantang dengan Materi Pengayaan",
            description=f"Performa sangat baik di semua aspek — siswa siap untuk tantangan yang lebih tinggi.",
            action="Berikan soal pengayaan atau proyek mandiri yang menantang agar potensi optimalnya terus berkembang."
        ))
    else:
        recs.append(RecommendationItem(
            title="Lakukan Check-in Rutin Setiap Dua Minggu",
            description=f"Pemantauan berkala membantu mendeteksi perubahan kondisi sebelum menjadi masalah lebih besar.",
            action="Luangkan 5 menit bicara santai dengan siswa setiap dua minggu — tanyakan hambatan dan perkembangannya."
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
