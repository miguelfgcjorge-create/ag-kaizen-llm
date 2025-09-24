from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
from typing import List
import yaml
import re

app = FastAPI(title="Ag-Kaizen-LLM", version="0.2.0")

# -------- Pydantic schema (enforces your JSON contract) --------
class Recommendation(BaseModel):
    action: str
    impact: str  # "low" | "medium" | "high"
    effort: str  # "low" | "medium" | "high"

class AnalysisResponse(BaseModel):
    summary: str
    flow: str  # enum enforced at runtime below
    wastes: List[str]
    root_causes: List[str]
    recommendations: List[Recommendation]
    quick_test: str
    kpis: List[str]
    next_check_in_days: int = Field(ge=1, le=90)

# -------- Load taxonomy on startup --------
with open("configs/taxonomy.yaml", "r", encoding="utf-8") as f:
    TAX = yaml.safe_load(f)

VALID_FLOWS = set(TAX["flows"])
VALID_WASTES = set(TAX["wastes"])
SYN = TAX["synonyms"]

# -------- Simple keyword-based classifier (placeholder for LLM/RAG) --------
def detect_flow(text: str) -> str:
    t = text.lower()
    for flow, words in SYN.items():
        if flow in VALID_FLOWS:
            if any(w in t for w in words):
                return flow
    # fallback guesses
    if "harvest" in t or "brown" in t or "cool" in t:
        return "post_harvest"
    return "field_ops"

def detect_wastes(text: str) -> List[str]:
    t = text.lower()
    hits = []
    for waste in VALID_WASTES:
        words = SYN.get(waste, [])
        if any(w in t for w in words):
            hits.append(waste)
    if not hits:
        # minimal sensible default
        if "truck" in t or "delay" in t or "late" in t:
            hits = ["waiting"]
        else:
            hits = ["motion"]
    return hits[:3]

# -------- API models --------
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

    flow_guess = detect_flow(text)
    wastes_guess = detect_wastes(text)

    # naive root causes and recs (replace with LLM+RAG later)
    root_causes = ["unverified_root_cause"]
    recs = [Recommendation(action="Run a 1-week PDCA pilot on one plot", impact="medium", effort="low")]

    # KPIs heuristic by flow
    if flow_guess == "post_harvest":
        kpis = ["time_to_cool_min", "storage_loss_pct", "claim_rate_pct"]
    elif flow_guess == "inputs_logistics":
        kpis = ["avg_steps_per_worker", "crates_per_hour"]
    else:
        kpis = ["cycle_time_min", "throughput_units_per_hour"]

    # Build analysis object
    analysis = AnalysisResponse(
        summary="Preliminary diagnosis (rules-only).",
        flow=flow_guess if flow_guess in VALID_FLOWS else "field_ops",
        wastes=[w for w in wastes_guess if w in VALID_WASTES] or ["motion"],
        root_causes=root_causes,
        recommendations=recs,
        quick_test="Pick one small change, measure daily, review after 7 days.",
        kpis=kpis,
        next_check_in_days=7
    )

    # Final validation: enforce flow & wastes enums manually
    if analysis.flow not in VALID_FLOWS:
        analysis.flow = "field_ops"
    analysis.wastes = [w for w in analysis.wastes if w in VALID_WASTES] or ["motion"]

    return {
        "reply": (
            "Diagnosis (prototype): I mapped your text to a flow and waste(s). "
            "Next we’ll plug in retrieval + a proper LLM to refine causes and actions."
        ),
        "analysis": analysis.model_dump()
    }
