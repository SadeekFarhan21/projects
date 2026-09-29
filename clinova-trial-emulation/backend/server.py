#!/usr/bin/env python3
"""
FastAPI server for Virtual Clinical Trial Agent
Exposes the agent pipeline as REST API endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import json
import os
from datetime import datetime
from pathlib import Path

from agent.main import run_pipeline

app = FastAPI(
    title="Virtual Clinical Trial API",
    description="API for running virtual clinical trials using AI agents",
    version="1.0.0",
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],  # Vite ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for run status (use Redis/DB in production)
run_status: Dict[str, dict] = {}


class TrialRequest(BaseModel):
    """Request model for creating a trial"""
    question: str
    description: Optional[str] = None


class TrialResponse(BaseModel):
    """Response model for trial creation"""
    run_id: str
    status: str
    message: str


class TrialStatusResponse(BaseModel):
    """Response model for trial status"""
    run_id: str
    status: str
    question: str
    created_at: str
    completed_at: Optional[str] = None
    run_folder: Optional[str] = None
    error: Optional[str] = None


class TrialResultResponse(BaseModel):
    """Response model for trial results"""
    run_id: str
    question: str
    causal_question: Optional[str] = None
    design_spec: Optional[str] = None
    code: Optional[str] = None
    omop_mappings: Optional[str] = None
    validator_feedback: Optional[str] = None


def run_trial_background(run_id: str, question: str):
    """Background task to run the trial pipeline"""
    try:
        run_status[run_id]["status"] = "running"

        # Run the pipeline
        run_folder = run_pipeline(question)

        # Update status
        run_status[run_id].update({
            "status": "completed",
            "run_folder": run_folder,
            "completed_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        run_status[run_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        })


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Virtual Clinical Trial API",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.post("/api/trials", response_model=TrialResponse)
async def create_trial(request: TrialRequest, background_tasks: BackgroundTasks):
    """
    Create and start a new virtual clinical trial

    The trial runs asynchronously in the background.
    Use the returned run_id to check status and retrieve results.
    """
    # Generate unique run ID
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Initialize run status
    run_status[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "question": request.question,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Start background task
    background_tasks.add_task(run_trial_background, run_id, request.question)

    return TrialResponse(
        run_id=run_id,
        status="queued",
        message="Trial started successfully. Use /api/trials/{run_id}/status to check progress.",
    )


@app.get("/api/trials/{run_id}/status", response_model=TrialStatusResponse)
async def get_trial_status(run_id: str):
    """Get the status of a running or completed trial"""
    if run_id not in run_status:
        raise HTTPException(status_code=404, detail="Trial not found")

    return TrialStatusResponse(**run_status[run_id])


@app.get("/api/trials/{run_id}/results", response_model=TrialResultResponse)
async def get_trial_results(run_id: str):
    """Get the results of a completed trial"""
    if run_id not in run_status:
        raise HTTPException(status_code=404, detail="Trial not found")

    trial = run_status[run_id]

    if trial["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Trial is {trial['status']}. Results only available for completed trials.",
        )

    run_folder = trial["run_folder"]

    # Read all output files
    def read_file(filename: str) -> Optional[str]:
        path = Path(run_folder) / filename
        if path.exists():
            return path.read_text()
        return None

    return TrialResultResponse(
        run_id=run_id,
        question=trial["question"],
        causal_question=read_file("step1_causal_question.txt"),
        design_spec=read_file("step2_design_spec.md"),
        code=read_file("step3_code.py"),
        omop_mappings=read_file("step2c_omop_mappings.md"),
        validator_feedback=read_file("step2_validator_feedback.txt"),
    )


@app.get("/api/trials", response_model=List[TrialStatusResponse])
async def list_trials():
    """List all trials"""
    return [TrialStatusResponse(**trial) for trial in run_status.values()]


@app.delete("/api/trials/{run_id}")
async def delete_trial(run_id: str):
    """Delete a trial from the registry"""
    if run_id not in run_status:
        raise HTTPException(status_code=404, detail="Trial not found")

    del run_status[run_id]
    return {"message": f"Trial {run_id} deleted successfully"}


@app.get("/api/examples")
async def get_example_trials():
    """Get list of example trials with their data"""
    examples_dir = Path(__file__).parent / "run" / "example-for-website"

    if not examples_dir.exists():
        return []

    examples = []
    for trial_dir in examples_dir.iterdir():
        if not trial_dir.is_dir():
            continue

        # Look for JSON file
        json_files = list(trial_dir.glob("*.json"))
        if not json_files:
            continue

        json_file = json_files[0]

        try:
            with open(json_file, 'r') as f:
                trial_data = json.load(f)

            # Read Python code if exists
            code_file = trial_dir / "trial_code.py"
            code = code_file.read_text() if code_file.exists() else None

            # Get image files
            images = [img.name for img in trial_dir.glob("*.png")]

            examples.append({
                "id": trial_dir.name,
                "name": trial_data.get("trial_config", {}).get("trial_name", trial_dir.name),
                "data": trial_data,
                "code": code,
                "images": images
            })
        except Exception as e:
            print(f"Error loading {trial_dir.name}: {e}")
            continue

    return examples


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
