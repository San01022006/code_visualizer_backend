# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import multiprocessing as mp
import time
import traceback

from mini_interpreter import MiniInterpreter, StepEvent

app = FastAPI()


class RunRequest(BaseModel):
    code: str
    timeout: float = 5.0   # seconds


def worker_run(code: str, out_q: mp.Queue):
    """
    Worker process: runs the MiniInterpreter in a separate process so that
    infinite loops or very long executions can be killed after a timeout.
    """
    try:
        interp = MiniInterpreter(code)
        interp.runAll()
        # Convert StepEvent dataclass objects to dicts
        steps = []
        for s in interp.steps:
            steps.append({
                "lineNo": s.lineNo,
                "codeLine": s.codeLine,
                "desc": s.desc,
                "varsSnapshot": s.varsSnapshot,
                "outputs": s.outputs,
                "visualizations": s.visualizations
            })
        out_q.put({"status": "ok", "steps": steps})
    except Exception as e:
        out_q.put({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        })


@app.post("/run")
def run_code(req: RunRequest):
    """
    Execute the submitted code using the MiniInterpreter inside a separate
    process with a time limit given by req.timeout (seconds).
    """
    manager = mp.Manager()
    q = manager.Queue()

    p = mp.Process(target=worker_run, args=(req.code, q))
    p.start()
    p.join(req.timeout)

    if p.is_alive():
        # Timeout reached, kill the worker
        p.terminate()
        p.join()
        return {
            "status": "timeout",
            "message": f"Execution exceeded {req.timeout} seconds and was terminated."
        }

    if q.empty():
        return {
            "status": "error",
            "message": "No result returned (interpreter crashed?)"
        }

    result = q.get()
    return result


@app.get("/health")
def health_check():
    """
    Simple health endpoint so the Android app or hosting platform
    can verify that the server is up without running any code.
    """
    return {"status": "ok"}


# Optional: local development entry point.
# In production (cloud hosting), the platform will usually run
# `uvicorn server:app` or similar, so this block is only for manual runs.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
