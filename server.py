# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import multiprocessing as mp
import time
import traceback
import io
import sys
import contextlib
import json

from mini_interpreter import MiniInterpreter, StepEvent

app = FastAPI()


class RunRequest(BaseModel):
    code: str
    timeout: float = 5.0   # seconds


def worker_run(code: str, out_q: mp.Queue):
    """
    Worker process: executes REAL Python code with exec(),
    captures stdout and variables, and returns them
    in a 'steps' list compatible with your current client.
    """
    try:
        # Isolated execution environment
        global_ns: Dict[str, Any] = {"__builtins__": __builtins__}
        local_ns: Dict[str, Any] = {}

        buf = io.StringIO()

        # Capture print() output
        with contextlib.redirect_stdout(buf):
            exec(code, global_ns, local_ns)

        stdout_text = buf.getvalue()
        output_lines = stdout_text.splitlines()

        # Collect variables (exclude internals)
        raw_vars: Dict[str, Any] = {
            k: v for k, v in local_ns.items() if not k.startswith("__")
        }

        # Make values JSON-serializable
        vars_snapshot: Dict[str, Any] = {}
        for k, v in raw_vars.items():
            try:
                json.dumps(v)  # test JSON serializability
                vars_snapshot[k] = v
            except TypeError:
                vars_snapshot[k] = repr(v)

        # Build a single "step" like your old structure
        step = {
            "lineNo": 0,
            "codeLine": "",
            "desc": "Execution finished",
            "varsSnapshot": vars_snapshot,
            "outputs": output_lines,
            "visualizations": []
        }

        out_q.put({"status": "ok", "steps": [step]})

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
