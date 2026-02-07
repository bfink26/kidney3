"""
Health Universe A2A Agent: Kidney Transportation + Recipient Recommendation

Agent Name:
    Kidney Transportation Option to Maximize Viability Giving Recipient Accessibility

What it does:
- Parses a kidney intake form (free text or JSON)
- Loads recipient match-run CSV + pump inventory/status CSV from the thread
- Scores recipients (ABO compatibility, wait time/list priority, logistics, EPTS suitability)
- Recommends a preservation method (SCS vs hypothermic machine perfusion) + pump options
- Retrieves quick evidence pointers from PubMed + OpenFDA
- Produces a Markdown report and a simple PDF artifact

Run with:
    uv run python main.py

Your agent will be available at http://localhost:8000
"""

from __future__ import annotations

import base64

import io
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import requests
import uvicorn
from openai import OpenAI
from sklearn.preprocessing import MinMaxScaler

from health_universe_a2a import Agent, AgentContext, create_app

from a2a.types import (
    AgentExtension,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
)

from health_universe_a2a.context import StreamingContext
from health_universe_a2a.types.extensions import (
    FILE_ACCESS_EXTENSION_URI,
    HU_LOG_LEVEL_EXTENSION_URI,
    UpdateImportance,
)
from health_universe_a2a.types.validation import ValidationRejected


# Configure poppler path to enable working with PDF files
# Comment this out as well as the poppler dependencies in packages.txt if you don't need PDFs
poppler_path = os.getenv("POPPLER_PATH")
if not poppler_path:
    # Common locations for poppler-utils
    common_paths = [
        "/usr/bin",  # Linux (apt install poppler-utils)
        "/opt/homebrew/bin",  # macOS Apple Silicon (brew install poppler)
        "/usr/local/bin",  # macOS Intel (brew install poppler)
        "/app/.apt/usr/bin",  # Heroku buildpack
    ]

    for path in common_paths:
        pdftoppm_path = os.path.join(path, "pdftoppm")
        if os.path.exists(pdftoppm_path):
            poppler_path = path
            break

if poppler_path:
    print(f"Using poppler from: {poppler_path}")
    # Add to PATH so pdf2image can find it
    os.environ["PATH"] = f"{poppler_path}:{os.environ.get('PATH', '')}"
else:
    print("WARNING: poppler not found. PDF processing may fail.")
    print("Install with: apt-get install poppler-utils (Linux) or brew install poppler (macOS)")




def _b64(data: bytes) -> str:
    """Base64-encode bytes for A2A FileWithBytes."""
    return base64.b64encode(data).decode('utf-8')


# ---------- Data models ----------


@dataclass
class KidneyIntake:
    raw_text: str

    # Identity
    case_id: str | None = None
    opo_id: str | None = None
    organ: str | None = None
    laterality: str | None = None

    # Compatibility essentials
    donor_abo: str | None = None
    donor_type: str | None = None  # DCD / DBD
    intended_abo_rule: str | None = None

    # Ischemia + timestamps
    wls_time: datetime | None = None
    asystole_time: datetime | None = None
    cold_flush_time: datetime | None = None
    cross_clamp_time: datetime | None = None
    packaged_time: datetime | None = None
    pickup_time_est: datetime | None = None
    warm_ischemia_functional_min: int | None = None
    warm_ischemia_total_min: int | None = None

    # Donor snapshot
    donor_age_years: int | None = None
    sex: str | None = None
    bmi: float | None = None
    cause_of_death: str | None = None
    comorbidities: dict[str, bool] = field(default_factory=dict)

    # Function signals
    creatinine_baseline: float | None = None
    creatinine_peak: float | None = None
    creatinine_last: float | None = None
    creatinine_trend: str | None = None
    urine_output_ml_kg_hr: float | None = None
    aki_noted: bool | None = None

    # Biopsy (if any)
    biopsy_done: bool | None = None
    glomerulosclerosis_pct: float | None = None
    ifta: str | None = None
    vascular_changes: str | None = None

    # Preservation status
    preservation_method: str | None = None
    solution: str | None = None
    storage_temp_c: str | None = None

    # Logistics
    location: str | None = None
    transport_options: str | None = None
    or_delay_risk: str | None = None



# ---------- Streaming compatibility context ----------

class _StreamingCompatContext:
    """
    Compatibility layer that provides the BackgroundContext-style interface
    (update_progress, add_artifact, document_client) on top of the A2A streaming
    context (SSE TaskUpdater).

    This allows the agent to run *synchronously* (no background job extension)
    while still:
      - emitting progress/status updates
      - attaching artifacts
      - reading/writing thread documents via DocumentClient
    """

    def __init__(self, streaming: StreamingContext):
        self._streaming = streaming
        self.user_id = streaming.user_id
        self.thread_id = streaming.thread_id
        self.file_access_token = streaming.file_access_token
        self.auth_token = streaming.auth_token
        self.metadata = streaming.metadata
        self.extensions = streaming.extensions
        self.updater = streaming.updater
        self.request_context = streaming.request_context
        self._documents = None
        # Ensures the task stream reaches a terminal state (completed/failed)
        self._terminal_sent = False

    @property
    def document_client(self):
        # Lazy import to keep module import-time light.
        if self._documents is None:
            from health_universe_a2a.documents import DocumentClient

            if not self.thread_id or not self.file_access_token:
                raise ValueError(
                    "Missing file access context (thread_id or file_access_token). "
                    "Ensure the FILE_ACCESS extension is enabled and the request includes it."
                )

            self._documents = DocumentClient(
                base_url=os.getenv("HU_NESTJS_URL", "https://apps.healthuniverse.com/api/v1"),
                access_token=self.file_access_token,
                thread_id=self.thread_id,
            )
        return self._documents

    async def close(self) -> None:
        if self._documents is not None:
            await self._documents.close()

    async def update_progress(
        self,
        message: str,
        progress: float | None = None,
        status: str = "working",
        importance: UpdateImportance = UpdateImportance.NOTICE,
    ) -> None:
        """
        Emit a status update over the task event stream.

        We include progress + importance in event metadata so UIs can render it
        if they choose to.
        """
        # Stream non-terminal status updates. The SDK will emit the terminal
        # completion state after handle_request() returns.
        normalized = (status or "working").strip().lower()
        meta: dict[str, Any] = {"task_status": normalized, HU_LOG_LEVEL_EXTENSION_URI: {"importance": importance.value}}
        if progress is not None:
            meta["progress"] = float(progress)

        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[Part(root=TextPart(text=message))],
        )
        await self.updater.update_status(state=TaskState.working, message=msg, final=False, metadata=meta)

    async def add_artifact(
        self,
        name: str,
        content: str | bytes,
        data_type: str = "text/plain",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach an artifact to the task stream."""
        artifact_meta: dict[str, Any] = {
            "data_type": data_type,
        }
        if description:
            artifact_meta["description"] = description
        if metadata:
            artifact_meta["metadata"] = metadata

        # Text artifacts
        if isinstance(content, str):
            parts = [Part(root=TextPart(text=content))]
        else:
            # Binary artifacts as FilePart
            parts = [
                Part(
                    root=FilePart(
                        file=FileWithBytes(bytes=_b64(content), mime_type=data_type, name=name),
                    )
                )
            ]

        await self.updater.add_artifact(
            parts=parts,
            name=name,
            metadata=artifact_meta,
            last_chunk=True,
        )



# ---------- Agent ----------


class MyAgent(Agent):
    """Kidney transportation and recipient recommendation agent."""

    def get_agent_name(self) -> str:
        return "Kidney Transportation Option to Maximize Viability Giving Recipient Accessibility"

    def get_agent_description(self) -> str:
        return (
            "Given kidney intake details plus recipient match-run and pump inventory CSVs, "
            "recommends a best-fit recipient and a preservation/transport strategy to maximize "
            "viability while accounting for logistics."
        )

    # --- Capabilities / extensions ---
    #
    # The stock `Agent` (alias of AsyncAgent) expects the Health Universe background-job
    # extension for long-running async processing. In some deployments that extension
    # is not available, which results in:
    #   "Background job extension missing - cannot process async"
    #
    # To make the agent work everywhere, we run in *streaming synchronous* mode:
    # - no background jobs (no missing extension)
    # - progress + artifacts streamed over the task event channel

    def supports_streaming(self) -> bool:
        return True

    def supports_push_notifications(self) -> bool:
        # Disable background-job execution (POST callbacks) and keep everything in-request.
        return False

    def get_extensions(self) -> list[AgentExtension]:
        # Advertise only what we actually need.
        return [
            AgentExtension(uri=FILE_ACCESS_EXTENSION_URI),
            AgentExtension(uri=HU_LOG_LEVEL_EXTENSION_URI),
        ]

    async def handle_request(
        self, message: str, context: StreamingContext, metadata: dict[str, Any]
    ) -> str | None:
        """
        Synchronous streaming handler:
        - validate
        - process in-request (no background jobs)
        - stream progress + artifacts via TaskUpdater
        """
        validation_result = await self.validate_message(message, metadata)

        if isinstance(validation_result, ValidationRejected):
            self.logger.warning(f"Message validation failed: {validation_result.reason}")
            msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(text=f"Validation failed: {validation_result.reason}"))],
            )
            await context.updater.reject(message=msg)
            return None

        # Start work (keeps stream open)
        await context.updater.start_work(
            message=Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(text="Starting work..."))],
            )
        )

        compat = _StreamingCompatContext(context)
        try:
            # Run the work synchronously in the request stream.
            # When this returns, the SDK will emit a TaskState.completed terminal event.
            result = await self.process_message(message, compat)  # type: ignore[arg-type]
            return result
        except Exception as e:
            # Ensure the stream reaches a terminal state on errors.
            fail_msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(text=f"Task failed: {e}"))],
            )
            await context.updater.failed(message=fail_msg)
            return None

        finally:
            # Normal path cleanup
            await compat.close()

    async def process_message(self, message: str, context: AgentContext) -> str:
        await context.update_progress("Parsing kidney intake...", progress=0.1)
        kidney = self._parse_kidney_intake(message)

        await context.update_progress("Loading recipient and pump CSVs...", progress=0.25)
        recipients_df, pumps_df, load_notes = await self._load_reference_data(context)

        await context.update_progress("Scoring kidney viability and logistics...", progress=0.4)
        viability = self._score_kidney_viability(kidney)

        await context.update_progress("Ranking candidate recipients...", progress=0.55)
        recipient_recommendation = self._rank_recipients(recipients_df, kidney, viability)

        await context.update_progress("Recommending preservation + transport...", progress=0.7)
        preservation_recommendation = self._recommend_preservation(kidney, pumps_df, recipient_recommendation)

        await context.update_progress("Fetching evidence (PubMed + OpenFDA)...", progress=0.8)
        evidence = self._fetch_evidence_snippets(kidney)

        await context.update_progress("Drafting final narrative (LLM)...", progress=0.9)
        narrative = self._generate_llm_summary(
            message=message,
            kidney=kidney,
            viability=viability,
            recipient_recommendation=recipient_recommendation,
            preservation_recommendation=preservation_recommendation,
            evidence=evidence,
            load_notes=load_notes,
        )

        report_markdown = self._build_markdown_report(
            kidney=kidney,
            viability=viability,
            load_notes=load_notes,
            recipient_recommendation=recipient_recommendation,
            preservation_recommendation=preservation_recommendation,
            evidence=evidence,
            narrative=narrative,
        )

        report_pdf = self._render_pdf(report_markdown)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"kidney-recommendation-{timestamp}.pdf"

        # Save into thread documents + expose as artifacts
        await context.document_client.write(
            name="Kidney Recommendation Report",
            content=report_pdf,
            filename=filename,
        )
        await context.add_artifact(
            name="Kidney Recommendation Report (PDF)",
            content=report_pdf,
            data_type="application/pdf",
        )
        await context.add_artifact(
            name="Kidney Recommendation (Markdown)",
            content=report_markdown,
            data_type="text/markdown",
        )

        await context.update_progress("Complete.", progress=1.0)
        return report_markdown

    # ---------- Parsing ----------

    def _parse_kidney_intake(self, message: str) -> KidneyIntake:
        raw_text = message.strip()

        # Support JSON payloads (nice for structured callers)
        if self._looks_like_json(raw_text):
            try:
                payload = json.loads(raw_text)
                if isinstance(payload, dict):
                    return self._parse_kidney_intake_from_json(payload, raw_text)
            except json.JSONDecodeError:
                pass

        # Fallback: parse free text intake form
        intake = KidneyIntake(raw_text=raw_text)

        # IDs / basics
        intake.case_id = self._re_group(raw_text, r"(?im)^\s*case\s*id\s*:\s*(.+?)\s*$")
        intake.opo_id = self._re_group(raw_text, r"(?im)^\s*opo[- ]id\s*:\s*(.+?)\s*$")
        intake.organ = self._re_group(raw_text, r"(?im)^\s*organ\s*:\s*(.+?)\s*$")
        intake.laterality = self._re_group(raw_text, r"(?im)^\s*laterality\s*:\s*(.+?)\s*$")

        # Compatibility essentials
        intake.donor_abo = self._re_group(raw_text, r"(?im)^\s*donor\s*abo\s*:\s*([AOB]{1,2}\s*\+?|-?)")
        intake.donor_type = self._re_group(
            raw_text, r"(?im)^\s*donor\s*type\s*:\s*(DCD|DBD)\b"
        )
        intake.intended_abo_rule = self._re_group(
            raw_text, r"(?im)^\s*intended\s*recipient\s*abo\s*compatibility\s*:\s*(.+?)\s*$"
        )

        # Times + ischemia
        intake.wls_time = self._parse_dt(self._re_group(raw_text, r"(?im)^\s*withdrawal.*?:\s*(.+?)\s*$"))
        intake.asystole_time = self._parse_dt(self._re_group(raw_text, r"(?im)^\s*asystole.*?:\s*(.+?)\s*$"))
        intake.cold_flush_time = self._parse_dt(
            self._re_group(raw_text, r"(?im)^\s*start\s*cold\s*flush.*?:\s*(.+?)\s*$")
        )
        intake.cross_clamp_time = self._parse_dt(self._re_group(raw_text, r"(?im)^\s*cross[- ]clamp.*?:\s*(.+?)\s*$"))
        intake.packaged_time = self._parse_dt(self._re_group(raw_text, r"(?im)^\s*kidney\s*packaged.*?:\s*(.+?)\s*$"))
        intake.pickup_time_est = self._parse_dt(self._re_group(raw_text, r"(?im)^\s*pickup\s*time\s*estimate\s*:\s*(.+?)\s*$"))

        intake.warm_ischemia_functional_min = self._parse_int(
            self._re_group(raw_text, r"(?im)^\s*functional\s*warm\s*ischemia.*?:\s*(\d+)")
        )
        intake.warm_ischemia_total_min = self._parse_int(
            self._re_group(raw_text, r"(?im)^\s*total\s*warm\s*ischemia.*?:\s*(\d+)")
        )

        # Donor snapshot
        intake.donor_age_years = self._parse_int(self._re_group(raw_text, r"(?im)^\s*donor\s*age\s*:\s*(\d+)"))
        intake.sex = self._re_group(raw_text, r"(?im)^\s*sex\s*:\s*(male|female)\b")
        height_cm = self._parse_float(self._re_group(raw_text, r"(?im)^\s*height\s*/\s*weight\s*:\s*(\d+)\s*cm"))
        weight_kg = self._parse_float(self._re_group(raw_text, r"(?im)^\s*height\s*/\s*weight\s*:\s*\d+\s*cm\s*/\s*(\d+)\s*kg"))
        if height_cm and weight_kg:
            h_m = height_cm / 100.0
            intake.bmi = round(weight_kg / (h_m * h_m), 1)
        intake.cause_of_death = self._re_group(raw_text, r"(?im)^\s*cause\s*of\s*death\s*:\s*(.+?)\s*$")

        # Comorbidities
        intake.comorbidities["hypertension"] = self._yes_no(raw_text, r"(?im)^\s*hypertension\s*:\s*(yes|no)")
        intake.comorbidities["diabetes"] = self._yes_no(raw_text, r"(?im)^\s*diabetes\s*:\s*(yes|no)")
        intake.comorbidities["smoking"] = bool(self._re_group(raw_text, r"(?im)^\s*smoking\s*:\s*(.+?)\s*$"))
        intake.comorbidities["vasopressors"] = self._yes_no(raw_text, r"(?im)^\s*vasopressors\s*:\s*(yes|no)")
        intake.comorbidities["hypotension_shock"] = self._yes_no(
            raw_text, r"(?im)^\s*hypotension\/shock\s*episode\s*:\s*(yes|no)"
        )

        # Function signals
        intake.creatinine_baseline = self._parse_float(
            self._re_group(raw_text, r"(?im)^\s*baseline.*?:\s*([0-9.]+)\s*mg\/dL")
        )
        intake.creatinine_peak = self._parse_float(
            self._re_group(raw_text, r"(?im)^\s*peak\s*:\s*([0-9.]+)\s*mg\/dL")
        )
        intake.creatinine_last = self._parse_float(
            self._re_group(raw_text, r"(?im)^\s*last.*?:\s*([0-9.]+)\s*mg\/dL")
        )
        intake.creatinine_trend = self._re_group(raw_text, r"(?im)^\s*trend\s*:\s*(.+?)\s*$")
        intake.urine_output_ml_kg_hr = self._parse_float(
            self._re_group(raw_text, r"(?im)^\s*urine\s*output\s*:\s*.*?\(([\d.]+)\s*mL\/kg\/hr")
        )
        aki_text = self._re_group(raw_text, r"(?im)^\s*aki\s*noted.*?:\s*(yes|no)")
        intake.aki_noted = None if aki_text is None else aki_text.strip().lower() == "yes"

        # Biopsy
        biopsy_text = self._re_group(raw_text, r"(?im)^\s*biopsy\s*performed\s*:\s*(yes|no)")
        intake.biopsy_done = None if biopsy_text is None else biopsy_text.strip().lower() == "yes"
        intake.glomerulosclerosis_pct = self._parse_float(
            self._re_group(raw_text, r"(?im)^\s*glomerulosclerosis\s*:\s*([0-9.]+)\s*%")
        )
        intake.ifta = self._re_group(raw_text, r"(?im)^\s*interstitial.*?\(IFTA\)\s*:\s*(.+?)\s*$")
        intake.vascular_changes = self._re_group(raw_text, r"(?im)^\s*vascular\s*changes\s*:\s*(.+?)\s*$")

        # Preservation / logistics
        intake.preservation_method = self._re_group(raw_text, r"(?im)^\s*current\s*preservation\s*method\s*:\s*(.+?)\s*$")
        intake.solution = self._re_group(raw_text, r"(?im)^\s*solution\s*used\s*:\s*(.+?)\s*$")
        intake.storage_temp_c = self._re_group(raw_text, r"(?im)^\s*storage\s*temperature\s*:\s*(.+?)\s*$")

        intake.location = self._re_group(raw_text, r"(?im)^\s*current\s*location.*?:\s*(.+?)\s*$")
        intake.transport_options = self._re_group(raw_text, r"(?im)^\s*transport\s*options\s*:\s*(.+?)\s*$")
        intake.or_delay_risk = self._re_group(raw_text, r"(?im)^\s*known\s*or\s*delay\s*risk.*?:\s*(.+?)\s*$")

        return intake

    def _parse_kidney_intake_from_json(self, payload: dict[str, Any], raw_text: str) -> KidneyIntake:
        def g(keys: Iterable[str]) -> str | None:
            for k in keys:
                if k in payload and payload[k] is not None:
                    return str(payload[k])
            return None

        intake = KidneyIntake(raw_text=raw_text)
        intake.case_id = g(["case_id", "caseId"])
        intake.opo_id = g(["opo_id", "opoId", "opo"])
        intake.organ = g(["organ"])
        intake.laterality = g(["laterality"])
        intake.donor_abo = g(["donor_abo", "abo", "donor_blood_type"])
        intake.donor_type = g(["donor_type", "dcd_dbd"])
        intake.location = g(["location", "current_location", "organ_location"])
        intake.preservation_method = g(["preservation_method", "current_preservation"])
        intake.solution = g(["solution", "preservation_solution"])

        intake.warm_ischemia_functional_min = self._parse_int(g(["warm_ischemia_functional_min", "fwi_min"]))
        intake.warm_ischemia_total_min = self._parse_int(g(["warm_ischemia_total_min", "twi_min"]))

        intake.cross_clamp_time = self._parse_dt(g(["cross_clamp_time", "cross_clamp"]))
        intake.cold_flush_time = self._parse_dt(g(["cold_flush_time", "cold_flush"]))
        intake.pickup_time_est = self._parse_dt(g(["pickup_time_estimate", "pickup_time"]))

        intake.donor_age_years = self._parse_int(g(["donor_age", "donor_age_years"]))
        intake.sex = g(["sex"])
        intake.creatinine_baseline = self._parse_float(g(["creatinine_baseline"]))
        intake.creatinine_peak = self._parse_float(g(["creatinine_peak"]))
        intake.creatinine_last = self._parse_float(g(["creatinine_last"]))
        intake.creatinine_trend = g(["creatinine_trend"])
        return intake

    # ---------- Data loading ----------

    async def _load_reference_data(
        self, context: AgentContext
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
        """
        Loads recipient match-run and pump inventory CSVs.
        Preferred source: documents uploaded to the thread (via document_client).
        Fallback: bundled sample CSVs in the repo, or env vars.
        """
        notes: list[str] = []
        recipients_df: pd.DataFrame | None = None
        pumps_df: pd.DataFrame | None = None

        documents = await context.document_client.list_documents()
        for doc in documents:
            name = (doc.name or "").lower()
            filename = (doc.filename or "").lower()
            if not (name.endswith(".csv") or filename.endswith(".csv")):
                continue

            content = await context.document_client.download_text(doc.id)
            df = pd.read_csv(io.StringIO(content))
            csv_type = self._infer_csv_type(df, name=name, filename=filename)

            if csv_type == "recipients" and recipients_df is None:
                recipients_df = df
                notes.append(f"Loaded recipients CSV from thread: {doc.filename or doc.name}")
            elif csv_type == "pumps" and pumps_df is None:
                pumps_df = df
                notes.append(f"Loaded pump inventory CSV from thread: {doc.filename or doc.name}")

        # Optional fallback paths (useful for local dev)
        if recipients_df is None:
            path = os.getenv("RECIPIENTS_CSV_PATH")
            if path and os.path.exists(path):
                recipients_df = pd.read_csv(path)
                notes.append(f"Loaded recipients CSV from RECIPIENTS_CSV_PATH: {path}")

        if pumps_df is None:
            path = os.getenv("PUMPS_CSV_PATH")
            if path and os.path.exists(path):
                pumps_df = pd.read_csv(path)
                notes.append(f"Loaded pumps CSV from PUMPS_CSV_PATH: {path}")

        # Bundled samples
        if recipients_df is None:
            bundled = "match_run_sample.csv"
            if os.path.exists(bundled):
                recipients_df = pd.read_csv(bundled)
                notes.append("No recipients CSV uploaded; using bundled match_run_sample.csv.")

        if pumps_df is None:
            bundled = "pump_feed_sample.csv"
            if os.path.exists(bundled):
                pumps_df = pd.read_csv(bundled)
                notes.append("No pump CSV uploaded; using bundled pump_feed_sample.csv.")

        if not notes:
            notes.append(
                "No CSVs found. Upload a recipient match-run CSV and a pump inventory CSV for best results."
            )

        return recipients_df, pumps_df, notes

    def _infer_csv_type(self, df: pd.DataFrame, name: str, filename: str) -> str | None:
        cols = {c.strip().lower() for c in df.columns}

        # Pump feed tends to have pump_id/status/estimated_ready
        if {"pump_id", "status"}.issubset(cols) or ("pump" in name or "pump" in filename):
            return "pumps"

        # Match run tends to have candidate_* fields
        if any(c.startswith("candidate_") for c in cols) or ("match" in name or "match" in filename):
            return "recipients"

        # Generic: look for "recipient" or "candidate"
        if "recipient" in cols or "candidate_id" in cols:
            return "recipients"

        return None

    # ---------- Scoring / recommendations ----------

    def _score_kidney_viability(self, kidney: KidneyIntake) -> dict[str, Any]:
        """
        Simple heuristic viability score (0-100). This is not clinical policy, just a ranking aid.
        """
        score = 100.0
        notes: list[str] = []

        donor_type = (kidney.donor_type or "").strip().upper()
        if donor_type == "DCD":
            score -= 8
            notes.append("DCD donor: higher ischemic stress vs DBD.")

        age = kidney.donor_age_years
        if age is not None:
            if age >= 60:
                score -= 15
                notes.append("Donor age ≥60: increased risk of delayed graft function.")
            elif age >= 50:
                score -= 8
                notes.append("Donor age 50-59: modest risk increase.")

        wim = kidney.warm_ischemia_functional_min or kidney.warm_ischemia_total_min
        if wim is not None:
            if wim >= 30:
                score -= 18
                notes.append("Warm ischemia ≥30 min: meaningful risk for ischemic injury.")
            elif wim >= 20:
                score -= 10
                notes.append("Warm ischemia 20-29 min: elevated risk.")

        # Creatinine
        cr_last = kidney.creatinine_last
        if cr_last is not None:
            if cr_last >= 2.5:
                score -= 12
                notes.append("Last creatinine ≥2.5 mg/dL: higher AKI / DGF risk.")
            elif cr_last >= 1.8:
                score -= 6
                notes.append("Last creatinine 1.8-2.4 mg/dL: modest AKI signal.")

        if kidney.aki_noted is True:
            score -= 6
            notes.append("AKI noted by procurement team.")

        # Comorbidities
        if kidney.comorbidities.get("hypertension"):
            score -= 4
            notes.append("Donor HTN: mild risk increase.")
        if kidney.comorbidities.get("diabetes"):
            score -= 8
            notes.append("Donor diabetes: higher risk of chronic injury.")
        if kidney.comorbidities.get("hypotension_shock"):
            score -= 4
            notes.append("Hypotension/shock episode: may worsen AKI.")

        # Biopsy
        gs = kidney.glomerulosclerosis_pct
        if gs is not None:
            if gs >= 15:
                score -= 14
                notes.append("Glomerulosclerosis ≥15%: reduced functional nephron mass.")
            elif gs >= 10:
                score -= 8
                notes.append("Glomerulosclerosis 10-14%: moderate chronic injury.")
            elif gs >= 5:
                score -= 3
                notes.append("Glomerulosclerosis 5-9%: mild chronic injury.")

        if kidney.ifta:
            if "moderate" in kidney.ifta.lower():
                score -= 10
                notes.append("IFTA moderate: chronic injury signal.")
            elif "mild" in kidney.ifta.lower():
                score -= 4
                notes.append("IFTA mild: small chronic injury signal.")
            elif "severe" in kidney.ifta.lower():
                score -= 16
                notes.append("IFTA severe: strong chronic injury signal.")

        if kidney.vascular_changes and "mild" in kidney.vascular_changes.lower():
            score -= 2
            notes.append("Mild vascular changes on biopsy.")

        # Clamp
        score = max(0.0, min(100.0, score))
        band = "High" if score >= 80 else "Moderate" if score >= 60 else "Lower"
        return {"score": round(score, 1), "band": band, "notes": notes}

    def _rank_recipients(
        self, recipients_df: pd.DataFrame | None, kidney: KidneyIntake, viability: dict[str, Any]
    ) -> dict[str, Any]:
        if recipients_df is None or recipients_df.empty:
            return {
                "status": "no_recipients",
                "message": "No recipient/match-run CSV provided; unable to rank candidates.",
                "top_candidate": None,
                "top_table": None,
            }

        df = recipients_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        # Column discovery for the provided sample format (candidate_* fields)
        abo_col = self._first_existing_column(df, ["candidate_blood_type", "abo", "blood_type", "recipient_abo"])
        age_col = self._first_existing_column(df, ["candidate_age_years", "age", "recipient_age"])
        epts_col = self._first_existing_column(df, ["candidate_epts_percent", "epts_percent", "epts"])
        cpra_col = self._first_existing_column(df, ["candidate_cpra_percent", "cpra_percent", "cpra"])
        dialysis_col = self._first_existing_column(df, ["dialysis_start_date", "dialysis_start"])
        priority_col = self._first_existing_column(df, ["priority_flags", "priority", "flags"])
        city_col = self._first_existing_column(df, ["program_city", "city"])
        state_col = self._first_existing_column(df, ["program_state", "state"])
        zip_col = self._first_existing_column(df, ["program_zip", "zip"])

        # ABO compatibility filter
        donor_abo = (kidney.donor_abo or "").strip().upper()
        if donor_abo and abo_col:
            allowed = self._allowed_recipient_abos(donor_abo)
            df = df[df[abo_col].astype(str).str.upper().str.extract(r"([AOB]{1,2})", expand=False).isin(allowed)]

        if df.empty:
            return {
                "status": "no_compatible_recipients",
                "message": "No ABO-compatible recipients found in the CSV.",
                "top_candidate": None,
                "top_table": None,
            }

        # Feature engineering
        dialysis_days = None
        if dialysis_col:
            dialysis_days = self._days_since(df[dialysis_col])

        if dialysis_days is not None:
            df["dialysis_days"] = dialysis_days


        priority_flag = None
        if priority_col:
            s = df[priority_col].fillna("").astype(str).str.lower()
            priority_flag = s.str.contains("prior").astype(float)

        # Logistics: estimate distance/time by geocoding (best effort)
        distance_mi = self._estimate_distance_to_programs(
            kidney_location=kidney.location,
            program_city=df[city_col] if city_col else None,
            program_state=df[state_col] if state_col else None,
            program_zip=df[zip_col] if zip_col else None,
        )

        if distance_mi is not None:
            df["distance_mi"] = distance_mi


        # Suitability: for marginal kidneys, align with higher EPTS; for high quality, align with lower EPTS
        target_epts = 20.0 if viability["score"] >= 80 else 70.0 if viability["score"] < 60 else 50.0
        epts_alignment = None
        if epts_col:
            epts_alignment = -1.0 * (df[epts_col].fillna(50).astype(float) - target_epts).abs()

        # Assemble numeric features into a matrix (some may be missing)
        features: dict[str, pd.Series] = {}

        if dialysis_days is not None:
            features["dialysis_days"] = dialysis_days.fillna(0).astype(float)
        if priority_flag is not None:
            features["priority_flag"] = priority_flag
        if distance_mi is not None:
            features["distance_mi"] = distance_mi.fillna(distance_mi.max() if not distance_mi.empty else 0).astype(float)
        if epts_alignment is not None:
            features["epts_alignment"] = epts_alignment.astype(float)
        if cpra_col:
            # Higher CPRA can mean harder-to-match; slight positive weight
            features["cpra"] = df[cpra_col].fillna(0).astype(float)

        if not features:
            return {
                "status": "insufficient_columns",
                "message": "Recipient CSV did not contain recognizable columns to score.",
                "top_candidate": None,
                "top_table": None,
            }

        X = pd.DataFrame(features).fillna(0)

        # Normalize features to 0..1
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Weighting: marginal kidneys emphasize logistics more
        marginal = viability["score"] < 60
        weights = {
            "dialysis_days": 0.35,
            "priority_flag": 0.15,
            "distance_mi": -0.35 if marginal else -0.2,
            "epts_alignment": 0.2,
            "cpra": 0.1,
        }

        score = pd.Series(0.0, index=X_scaled.index)
        for col, w in weights.items():
            if col in X_scaled.columns:
                score += X_scaled[col] * float(w)

        ranked = df.copy()
        ranked["score"] = score
        ranked = ranked.sort_values("score", ascending=False)

        top_candidate = ranked.head(1).to_dict(orient="records")[0]
        top_table = ranked.head(10)

        return {
            "status": "ranked",
            "message": "Recipient ranking completed.",
            "top_candidate": top_candidate,
            "top_table": top_table,
            "scoring_explanation": {
                "kidney_viability_score": viability["score"],
                "marginal_kidney_logic": marginal,
                "target_epts_percent": target_epts,
                "feature_weights": weights,
            },
        }

    def _recommend_preservation(
        self,
        kidney: KidneyIntake,
        pumps_df: pd.DataFrame | None,
        recipient_recommendation: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Decide SCS vs HMP, and suggest specific pumps if available.
        """
        warm = kidney.warm_ischemia_functional_min or kidney.warm_ischemia_total_min or 0
        donor_type = (kidney.donor_type or "").strip().upper()
        base_method = kidney.preservation_method or "Static cold storage"

        # Estimate cold ischemia time (CIT) to implant (very rough, best effort)
        cit_est = self._estimate_cit_hours(kidney, recipient_recommendation)

        # Pump availability (and best pumps)
        pump_info = self._select_pumps(kidney, pumps_df, max_pumps=3)

        pump_available = bool(pump_info.get("suggested_pumps"))
        should_pump = (donor_type == "DCD") or (warm >= 20) or (cit_est is not None and cit_est >= 10)

        if should_pump and pump_available:
            method = "Hypothermic machine perfusion (HMP)"
            reason = (
                "Risk profile suggests benefit from pump perfusion (DCD and/or elevated ischemia forecast). "
                "A pump appears available in inventory."
            )
        elif should_pump and not pump_available:
            method = "Static cold storage (SCS)"
            reason = (
                "Pump perfusion would likely help, but no AVAILABLE pump was found in the provided feed. "
                "Continue SCS and minimize cold ischemia time."
            )
        else:
            method = base_method
            reason = "Current risk profile does not strongly indicate pump perfusion; maintain SCS and optimize logistics."

        return {
            "suggested_method": method,
            "reason": reason,
            "estimated_cit_hours": cit_est,
            "pump_availability": pump_info,
        }

    def _select_pumps(self, kidney: KidneyIntake, pumps_df: pd.DataFrame | None, max_pumps: int) -> dict[str, Any]:
        if pumps_df is None or pumps_df.empty:
            return {"status": "no_pump_csv", "suggested_pumps": []}

        df = pumps_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        status_col = self._first_existing_column(df, ["status", "available", "in_service"])
        ready_col = self._first_existing_column(df, ["estimated_ready_minutes", "ready_minutes"])
        city_col = self._first_existing_column(df, ["city"])
        state_col = self._first_existing_column(df, ["state"])
        zip_col = self._first_existing_column(df, ["zip"])
        pump_id_col = self._first_existing_column(df, ["pump_id", "id"])

        if not status_col or not pump_id_col:
            return {"status": "unrecognized_pump_schema", "suggested_pumps": []}

        available_mask = df[status_col].astype(str).str.contains(r"available|yes|true", case=False, regex=True)
        available = df[available_mask].copy()

        if available.empty:
            return {"status": "no_available_pumps", "suggested_pumps": []}

        # Compute distance to kidney location (best effort)
        distance_mi = self._estimate_distance_to_programs(
            kidney_location=kidney.location,
            program_city=available[city_col] if city_col else None,
            program_state=available[state_col] if state_col else None,
            program_zip=available[zip_col] if zip_col else None,
        )
        if distance_mi is not None:
            available["distance_mi"] = distance_mi

        if ready_col:
            available["estimated_ready_minutes"] = available[ready_col].fillna(9999).astype(float)
        else:
            available["estimated_ready_minutes"] = 9999.0

        # Rank: ready sooner and closer
        sort_cols = ["estimated_ready_minutes"]
        if "distance_mi" in available.columns:
            sort_cols.append("distance_mi")

        suggested = available.sort_values(sort_cols, ascending=True).head(max_pumps)

        keep_cols = [c for c in suggested.columns if c in {pump_id_col, status_col, "estimated_ready_minutes", "distance_mi", "city", "state", "zip"}]
        suggested_records = suggested[keep_cols].to_dict(orient="records")

        return {"status": "ok", "suggested_pumps": suggested_records}

    # ---------- Evidence (PubMed + OpenFDA) ----------

    def _fetch_evidence_snippets(self, kidney: KidneyIntake) -> list[dict[str, str]]:
        snippets: list[dict[str, str]] = []

        # PubMed (no API key needed)
        try:
            query = "kidney transplant hypothermic machine perfusion DCD warm ischemia"
            snippets.extend(self._query_pubmed(query=query, retmax=3))
        except requests.RequestException:
            pass

        # OpenFDA (no API key needed): pull safety/label pointers for common peri-procurement drugs
        try:
            # If vasopressors were used, "vasopressin" is a common agent; otherwise default to heparin.
            terms = ["heparin"]
            if kidney.comorbidities.get("vasopressors"):
                terms.append("vasopressin")
            for term in terms:
                snippets.extend(self._query_openfda_label(term=term, limit=1))
        except requests.RequestException:
            pass

        return snippets

    def _query_openfda_label(self, term: str, limit: int) -> list[dict[str, str]]:
        """
        OpenFDA drug label endpoint. Returns short pointers (not full text).
        """
        url = "https://api.fda.gov/drug/label.json"
        # Search generic name first; fallback to any text search if needed
        searches = [
            f'openfda.generic_name:"{term}"',
            f'openfda.brand_name:"{term}"',
            term,
        ]

        for s in searches:
            resp = requests.get(url, params={"search": s, "limit": limit}, timeout=10)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("results"):
                label = data["results"][0]
                openfda = label.get("openfda", {})
                brand = (openfda.get("brand_name") or [term])[0]
                generic = (openfda.get("generic_name") or [term])[0]
                purpose = (label.get("purpose") or [""])[0]
                # Provide a link to the API query for traceability
                api_link = f"{url}?search={requests.utils.quote(s)}&limit={limit}"
                title = f"OpenFDA label: {brand} ({generic})"
                if purpose:
                    title += f" - {purpose[:80]}"
                return [{"title": title, "url": api_link}]
        return []

    def _generate_llm_summary(
        self,
        message: str,
        kidney: KidneyIntake,
        viability: dict[str, Any],
        recipient_recommendation: dict[str, Any],
        preservation_recommendation: dict[str, Any],
        evidence: list[dict[str, str]],
        load_notes: list[str],
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY not set; skipping LLM narrative. (Upload CSVs for deterministic scoring.)"

        # Model: user specified GPT-4; keep configurable.
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)

        evidence_summary = "\n".join(f"- {item['title']} ({item['url']})" for item in evidence) or "None"

        prompt = (
            "You are a transplant coordinator assistant. Use the provided kidney intake and computed "
            "rankings to produce a clear recommendation.\n\n"
            "Instructions:\n"
            "1) State the top recipient and why (ABO, wait time/priority, logistics, EPTS fit).\n"
            "2) State the recommended preservation method and transport plan.\n"
            "3) Call out major risks (ischemia, AKI, biopsy, donor comorbidities) and mitigation steps.\n"
            "4) Keep it concise and actionable.\n\n"
            f"Kidney viability score: {viability}\n\n"
            f"Load notes: {load_notes}\n\n"
            f"Recipient recommendation data: {recipient_recommendation}\n\n"
            f"Preservation recommendation data: {preservation_recommendation}\n\n"
            f"Evidence pointers: {evidence_summary}\n\n"
            f"Raw intake text:\n{message}\n"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert transplant allocation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    # ---------- Reporting ----------

    def _build_markdown_report(
        self,
        kidney: KidneyIntake,
        viability: dict[str, Any],
        load_notes: list[str],
        recipient_recommendation: dict[str, Any],
        preservation_recommendation: dict[str, Any],
        evidence: list[dict[str, str]],
        narrative: str,
    ) -> str:
        recipient = recipient_recommendation.get("top_candidate") or {}
        top_table = recipient_recommendation.get("top_table")

        recipient_lines = "\n".join(
            f"- **{k.replace('_', ' ').title()}**: {v}" for k, v in recipient.items()
        ) or "- (No recipient data available.)"

        evidence_lines = "\n".join(f"- [{item['title']}]({item['url']})" for item in evidence) or "- None"

        viability_notes = "\n".join(f"- {n}" for n in viability.get("notes", [])) or "- None"

        load_lines = "\n".join(f"- {n}" for n in load_notes) or "- None"

        top_table_md = ""
        if isinstance(top_table, pd.DataFrame) and not top_table.empty:
            view = top_table.copy()
            # Don't explode the report with huge tables
            keep = [c for c in view.columns if c in {"sequence_number", "candidate_id", "score", "candidate_blood_type", "candidate_epts_percent", "candidate_cpra_percent", "program_city", "program_state", "transplant_program"}]
            if keep:
                view = view[keep]
            top_table_md = "\n\n### Top 10 Candidates (scored)\n\n" + self._df_to_markdown(view, max_rows=10)

        cit = preservation_recommendation.get("estimated_cit_hours")
        cit_str = "Unknown" if cit is None else f"{cit:.1f} hours"

        pump_info = preservation_recommendation.get("pump_availability", {})
        pumps = pump_info.get("suggested_pumps") or []
        pumps_md = "\n".join(
            f"- **{p.get('pump_id', p.get('id'))}** | {p.get('status')} | ready "
            f"{int(p.get('estimated_ready_minutes', 0))} min | ~{p.get('distance_mi', '??')} mi"
            for p in pumps
        ) or "- None found / not provided."

        return (
            "# Kidney Allocation + Preservation Recommendation\n\n"
            "## Inputs Loaded\n"
            f"{load_lines}\n\n"
            "## Kidney Intake Summary (parsed)\n"
            f"- **Case / OPO**: {kidney.case_id or 'Unknown'} / {kidney.opo_id or 'Unknown'}\n"
            f"- **Organ**: {kidney.organ or 'Kidney'} {kidney.laterality or ''}\n"
            f"- **Donor ABO**: {kidney.donor_abo or 'Unknown'}\n"
            f"- **Donor Type**: {kidney.donor_type or 'Unknown'}\n"
            f"- **Warm ischemia (functional/total)**: "
            f"{kidney.warm_ischemia_functional_min or 'Unknown'} / {kidney.warm_ischemia_total_min or 'Unknown'} min\n"
            f"- **Cross-clamp time**: {kidney.cross_clamp_time or 'Unknown'}\n"
            f"- **Cold flush time**: {kidney.cold_flush_time or 'Unknown'}\n"
            f"- **Location**: {kidney.location or 'Unknown'}\n"
            f"- **Current preservation**: {kidney.preservation_method or 'Unknown'}\n"
            f"- **Solution**: {kidney.solution or 'Unknown'}\n\n"
            "## Viability Score (heuristic)\n"
            f"- **Score**: {viability['score']} / 100\n"
            f"- **Band**: {viability['band']}\n"
            "### Drivers\n"
            f"{viability_notes}\n\n"
            "## Recommended Recipient\n"
            f"{recipient_lines}\n"
            f"{top_table_md}\n\n"
            "## Preservation + Transport Recommendation\n"
            f"- **Suggested method**: {preservation_recommendation['suggested_method']}\n"
            f"- **Rationale**: {preservation_recommendation['reason']}\n"
            f"- **Estimated cold ischemia time (rough)**: {cit_str}\n\n"
            "### Suggested Pumps (if using HMP)\n"
            f"{pumps_md}\n\n"
            "## Evidence Pointers\n"
            f"{evidence_lines}\n\n"
            "## Narrative Summary\n"
            f"{narrative}\n"
        )

    # ---------- Logistics helpers ----------

    def _estimate_cit_hours(self, kidney: KidneyIntake, recipient_recommendation: dict[str, Any]) -> float | None:
        """
        Very rough CIT estimate:
        (cross clamp -> pickup estimate) + (travel time to center) + (OR delay)
        """
        if kidney.cross_clamp_time is None:
            return None

        base_hours = 0.0
        if kidney.pickup_time_est is not None:
            delta = (kidney.pickup_time_est - kidney.cross_clamp_time).total_seconds() / 3600.0
            base_hours += max(0.0, delta)

        # Travel time
        travel_hours = None
        top = recipient_recommendation.get("top_candidate") or {}
        # If scored table included distance_mi, use it; otherwise try city/state geocode
        dist = None
        for key in ("distance_mi", "distance_miles", "distance_km"):
            if key in top:
                try:
                    dist = float(top[key])
                except Exception:
                    dist = None
                break

        if dist is not None:
            travel_hours = dist / 55.0 + 0.5  # ground + handling
        else:
            # Can't infer
            travel_hours = None

        if travel_hours is not None:
            base_hours += max(0.0, travel_hours)

        # OR delay risk: crude adders
        if kidney.or_delay_risk:
            risk = kidney.or_delay_risk.lower()
            if "high" in risk:
                base_hours += 4.0
            elif "moderate" in risk:
                base_hours += 2.0
            elif "low" in risk:
                base_hours += 0.5

        return base_hours if base_hours > 0 else None

    def _estimate_distance_to_programs(
        self,
        kidney_location: str | None,
        program_city: pd.Series | None,
        program_state: pd.Series | None,
        program_zip: pd.Series | None,
    ) -> pd.Series | None:
        """
        Best-effort distance in miles from kidney_location (free text) to program city/state/zip.
        Uses OpenStreetMap Nominatim for geocoding. If geocoding fails, returns None.
        """
        if not kidney_location:
            return None
        if program_city is None and program_zip is None:
            return None

        origin = self._geocode_place(kidney_location)
        if origin is None:
            return None

        # Build destination strings
        if program_zip is not None:
            dest = program_zip.fillna("").astype(str)
        else:
            c = program_city.fillna("").astype(str)
            s = program_state.fillna("").astype(str) if program_state is not None else ""
            dest = (c + ", " + s).str.strip(", ")

        distances: list[float | None] = []
        for place in dest.tolist():
            if not place or place.strip() in {"nan", "none"}:
                distances.append(None)
                continue
            coord = self._geocode_place(place)
            if coord is None:
                distances.append(None)
                continue
            distances.append(self._haversine_miles(origin[0], origin[1], coord[0], coord[1]))

        return pd.Series(distances, index=dest.index, dtype="float")

    _geo_cache: dict[str, tuple[float, float]] = {}

    def _geocode_place(self, place: str) -> tuple[float, float] | None:
        place_key = place.strip().lower()
        if not place_key:
            return None
        if place_key in self._geo_cache:
            return self._geo_cache[place_key]

        # Allow disabling external geocoding (some deployments are offline)
        if os.getenv("DISABLE_GEOCODING", "false").lower() == "true":
            return None

        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "health-universe-a2a-kidney-agent/0.1"}
        params = {"q": place, "format": "json", "limit": 1}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            self._geo_cache[place_key] = (lat, lon)
            return lat, lon
        except Exception:
            return None

    @staticmethod
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 3958.7613  # Earth radius in miles
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        d1 = math.radians(lat2 - lat1)
        d2 = math.radians(lon2 - lon1)
        a = math.sin(d1 / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d2 / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    # ---------- PubMed ----------

    def _query_pubmed(self, query: str, retmax: int) -> list[dict[str, str]]:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax}
        response = requests.get(base, params=params, timeout=10)
        response.raise_for_status()
        ids = response.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_response = requests.get(
            summary_url,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=10,
        )
        summary_response.raise_for_status()
        summaries = summary_response.json().get("result", {})

        snippets: list[dict[str, str]] = []
        for pmid in ids:
            item = summaries.get(pmid)
            if not item:
                continue
            snippets.append({"title": item.get("title", "PubMed Article"), "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
        return snippets

    # ---------- Utilities ----------

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        t = text.strip()
        return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))

    @staticmethod
    def _re_group(text: str, pattern: str) -> str | None:
        m = re.search(pattern, text)
        if not m:
            return None
        return m.group(1).strip()

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(float(str(value).strip()))
        except Exception:
            return None

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except Exception:
            return None

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        v = value.strip()
        fmts = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%y %H:%M",
            "%m/%d/%y %H:%M:%S",
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                continue
        # Try ISO-8601
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return None

    @staticmethod
    def _yes_no(text: str, pattern: str) -> bool:
        m = re.search(pattern, text)
        if not m:
            return False
        return m.group(1).strip().lower() == "yes"

    @staticmethod
    def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _allowed_recipient_abos(donor_abo: str) -> set[str]:
        """
        Basic ABO rules for kidney allocation (Rh ignored).
        """
        abo = re.sub(r"[^A-Z]", "", donor_abo.upper())
        if abo == "O":
            return {"O"}
        if abo == "A":
            return {"A", "AB"}
        if abo == "B":
            return {"B", "AB"}
        if abo == "AB":
            return {"AB"}
        return {"A", "B", "AB", "O"}

    @staticmethod
    def _days_since(series: pd.Series) -> pd.Series:
        """
        Parse a date column and compute days from that date until today (UTC date).
        """
        today = pd.Timestamp.utcnow().normalize()
        parsed = pd.to_datetime(series, errors="coerce")
        return (today - parsed).dt.days


    
    @staticmethod
    def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
        """
        Render a small markdown table without optional dependencies (like tabulate).
        """
        if df.empty:
            return ""

        view = df.head(max_rows).copy()
        cols = list(view.columns)

        view = view.astype(str).replace("nan", "").replace("None", "")

        widths: dict[str, int] = {}
        for c in cols:
            widths[c] = max([len(c)] + [len(v) for v in view[c].tolist()])

        def row(values: list[str]) -> str:
            return "| " + " | ".join(v.ljust(widths[c]) for v, c in zip(values, cols)) + " |"

        header = row(cols)
        sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"

        body_rows: list[str] = []
        for _, r in view.iterrows():
            body_rows.append(row([str(r[c]) for c in cols]))

        return "\n".join([header, sep] + body_rows)



    def _render_pdf(self, markdown_text: str) -> bytes:
        """
        A tiny (monospace-ish) PDF renderer good enough for a report artifact.
        """
        lines = markdown_text.splitlines()
        content_lines = [self._escape_pdf_text(line) for line in lines if line.strip()]
        text_stream = [
            "BT",
            "/F1 10 Tf",
            "12 TL",
            "72 740 Td",
        ]
        # Keep within page: stop early if too long
        max_lines = 55
        for line in content_lines[:max_lines]:
            text_stream.append(f"({line}) Tj")
            text_stream.append("T*")
        if len(content_lines) > max_lines:
            text_stream.append("(...) Tj")
            text_stream.append("T*")
        text_stream.append("ET")
        content = "\n".join(text_stream)
        content_bytes = content.encode("latin-1", errors="replace")

        objects: list[bytes] = []
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objects.append(
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
            b"endobj\n"
        )
        objects.append(
            f"4 0 obj\n<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1")
            + content_bytes
            + b"\nendstream\nendobj\n"
        )
        objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

        xref_positions = []
        pdf_body = b"%PDF-1.4\n"
        for obj in objects:
            xref_positions.append(len(pdf_body))
            pdf_body += obj

        xref_start = len(pdf_body)
        xref_entries = [b"0000000000 65535 f \n"]
        for pos in xref_positions:
            xref_entries.append(f"{pos:010d} 00000 n \n".encode("latin-1"))
        xref_table = b"xref\n0 6\n" + b"".join(xref_entries)
        trailer = (
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
            + str(xref_start).encode("latin-1")
            + b"\n%%EOF"
        )
        return pdf_body + xref_table + trailer

    @staticmethod
    def _escape_pdf_text(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# Create the ASGI app
app = create_app(MyAgent())

if __name__ == "__main__":
    # Configuration from environment
    port = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8000")))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"

    # Run the server
    uvicorn.run(
        "main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
