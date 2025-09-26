from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import yaml, os, json, re

# --- OpenAI SDK ---
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Ag-Kaizen-LLM", version="0.3.0")

# -------- Pydantic schema (your contract) --------
class Recommendation(BaseModel):
    action: str
    impact: str  # "low" | "medium" | "high"
    effort: str  # "low" | "medium" | "high"

class AnalysisResponse(BaseModel):
    summary: str
    flow: str
    wastes: List[str]
    root_causes: List[str]
    recommendations: List[Recommendation]
    quick_test: str
    kpis: List[str]
    next_check_in_days: int = Field(ge=1, le=90)

# -------- Load taxonomy --------
with open("configs/taxonomy.yaml", "r", encoding="utf-8") as f:
    TAX = yaml.safe_load(f)
VALID_FLOWS = set(TAX["flows"])
VALID_WASTES = set(TAX["wastes"])
SYN = TAX["synonyms"]

# -------- Rules fallback (used if LLM fails) --------
def detect_flow(text: str) -> str:
    t = text.lower()
    for flow, words in SYN.items():
        if flow in VALID_FLOWS and any(w in t for w in words):
            return flow
    if any(w in t for w in ["harvest","brown","cool"]): return "post_harvest"
    return "field_ops"

def detect_wastes(text: str) -> List[str]:
    t = text.lower()
    hits = []
    for waste in VALID_WASTES:
        words = SYN.get(waste, [])
        if any(w in t for w in words): hits.append(waste)
    if not hits:
        hits = ["waiting"] if any(w in t for w in ["truck","delay","late"]) else ["motion"]
    return hits[:3]

def fallback_analysis(text: str) -> AnalysisResponse:
    flow_guess = detect_flow(text)
    wastes_guess = [w for w in detect_wastes(text) if w in VALID_WASTES] or ["motion"]
    kpis = {
        "post_harvest": ["time_to_cool_min","storage_loss_pct","claim_rate_pct"],
        "inputs_logistics":["avg_steps_per_worker","crates_per_hour"]
    }.get(flow_guess, ["cycle_time_min","throughput_units_per_hour"])
    return AnalysisResponse(
        summary="Preliminary diagnosis (rules-only).",
        flow=flow_guess if flow_guess in VALID_FLOWS else "field_ops",
        wastes=wastes_guess,
        root_causes=["unverified_root_cause"],
        recommendations=[Recommendation(action="Run a 1-week PDCA pilot on one plot", impact="medium", effort="low")],
        quick_test="Pick one small change, measure daily, review after 7 days.",
        kpis=kpis,
        next_check_in_days=7
    )

# -------- LLM assist --------
SYSTEM = """You are an Agriculture Kaizen Consultant.
Return a concise plan and ALWAYS end with a JSON block that matches this schema:
summary(str), flow(one of field_ops, post_harvest, livestock, inputs_logistics, back_office),
wastes(array), root_causes(array), recommendations(array of {action,impact,effort}),
quick_test(str), kpis(array), next_check_in_days(int 1-90).
Be concrete, farmer-friendly, no chemical dosages.
"""

FEWSHOT_USER = "Lettuce browns before delivery; trucks are every 2 days."
FEWSHOT_ASSISTANT = (
    "Diagnosis: post-harvest waiting/defects; add rapid pre-cool + daily micro-dispatch.\n"
    "```json\n"
    "{\n"
    "  \"summary\":\"Lettuce browning before delivery\",\n"
    "  \"flow\":\"post_harvest\",\n"
    "  \"wastes\":[\"waiting\",\"defects\"],\n"
    "  \"root_causes\":[\"delayed dispatch\",\"no rapid pre-cool\"],\n"
    "  \"recommendations\":[\n"
    "    {\"action\":\"Pre-cool within 90 min\",\"impact\":\"high\",\"effort\":\"medium\"},\n"
    "    {\"action\":\"Switch to smaller daily shipments\",\"impact\":\"high\",\"effort\":\"medium\"}\n"
    "  ],\n"
    "  \"quick_test\":\"Pilot pre-cool + daily dispatch on Lot A for 1 week\",\n"
    "  \"kpis\":[\"time_to_cool_min\",\"storage_loss_pct\",\"claim_rate_pct\"],\n"
    "  \"next_check_in_days\":7\n"
    "}\n"
    "```"
)

def try_llm(text: str) -> AnalysisResponse | None:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content": SYSTEM},
                {"role":"user","content": FEWSHOT_USER},
                {"role":"assistant","content": FEWSHOT_ASSISTANT},
                {"role":"user","content": text}
            ],
            temperature=0.2,
            max_tokens=800,
        )
        content = resp.choices[0].message.content
        m = re.findall(r"```json\\s*(\\{[\\s\\S]*?\\})\\s*```", content)
        raw = m[-1] if m else content.strip()
        data = json.loads(raw)
        if data.get("flow") not in VALID_FLOWS: data["flow"] = "field_ops"
        data["wastes"] = [w for w in data.get("wastes", []) if w in VALID_WASTES] or ["motion"]
        return AnalysisResponse(**data)
    except Exception:
        return None

# -------- API --------
class ChatRequest(BaseModel):
    user_text: str

@app.get("/")
def home():
    return {"message": "Ag-Kaizen-LLM is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    text = req.user_text.strip()
    if not text:
        return {"error": "Please describe your issue (e.g., irrigation delays, post-harvest spoilage)."}
    analysis = try_llm(text) or fallback_analysis(text)
    reply = "Kaizen diagnosis generated via LLM." if try_llm is not None else "Diagnosis (rules fallback)."
    return {"reply": reply, "analysis": analysis.model_dump()}
