# Kidney Transportation + Recipient Recommendation (A2A Agent)

This Health Universe **A2A** agent ingests a **kidney intake form** (free text or JSON) and recommends:

- The **best-fit recipient** from an uploaded **match-run / recipient list CSV**
- The **best preservation method** (Static Cold Storage vs Hypothermic Machine Perfusion)
- A short **transport/ischemia-risk narrative**, plus evidence pointers (PubMed + OpenFDA)

It returns a Markdown report and attaches a **PDF report** as an artifact.

## What to Upload (recommended)

Upload two CSVs in the conversation thread:

1. **Recipient / match-run CSV**
   - Example columns (supported): `candidate_id`, `candidate_blood_type`, `dialysis_start_date`,
     `candidate_epts_percent`, `candidate_cpra_percent`, `program_city`, `program_state`, `program_zip`

2. **Pump inventory / status CSV**
   - Example columns (supported): `pump_id`, `status`, `estimated_ready_minutes`, `city`, `state`, `zip`

The agent will also run locally using the bundled sample CSVs if you don't upload any.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | **Required** for LLM narrative | (none) |
| `OPENAI_MODEL` | Model name | `gpt-4o-mini` |
| `DISABLE_GEOCODING` | Disable OpenStreetMap geocoding if offline | `false` |
| `RECIPIENTS_CSV_PATH` | Local dev: path to recipient CSV | (none) |
| `PUMPS_CSV_PATH` | Local dev: path to pump CSV | (none) |
| `PORT` or `AGENT_PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `RELOAD` | Auto-reload (local dev) | `false` |

## Quick Start

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Run the agent**
   ```bash
   uv run python main.py
   ```

   For local dev with auto-reload:
   ```bash
   RELOAD=true uv run python main.py
   ```

3. **Test it**
   ```bash
   # View agent card
   curl http://localhost:8000/.well-known/agent.json

   # Send a message (paste a kidney intake form)
   curl -X POST http://localhost:8000/ \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "Donor ABO: O+\nDonor type: DCD\nFunctional warm ischemia: 24 minutes\nCurrent location: Manhattan, NY"}]}}}'
   ```

## Notes

- Recipient scoring and preservation recommendations are **heuristic decision aids**, not clinical policy.
- Geocoding (for distance estimates) uses OpenStreetMap Nominatim and can be disabled with `DISABLE_GEOCODING=true`.

## Documentation

- [SDK Documentation](https://github.com/Health-Universe/healthuniverse-a2a-sdk-python)
- [A2A Protocol](https://a2a.ai)


## Deployment note: avoiding the Background Job extension

Some Health Universe environments do **not** provide the Background Job extension metadata.
When an agent advertises background jobs, requests may log:

`Background job extension missing - cannot process async`

This agent is configured to run in **synchronous streaming mode**:

- **No background-job extension required**
- Progress updates + artifacts stream over the task event channel
- Documents still read/write via `context.document_client` (File Access extension)

If your environment *does* support background jobs and you want true detached async processing,
remove the overrides for `supports_push_notifications()`, `get_extensions()`, and `handle_request()`.
