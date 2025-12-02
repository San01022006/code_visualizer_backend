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
    but also traces each executed line to produce step-by-step
    data compatible with your visualizer (Prev / Next).
    """
    try:
        # Isolated execution environment
        global_ns: Dict[str, Any] = {"__builtins__": __builtins__}
        local_ns: Dict[str, Any] = {}

        buf = io.StringIO()
        code_lines = code.splitlines()

        # Compile user code with a known filename
        code_obj = compile(code, "<user_code>", "exec")

        steps = []
        outputs_so_far: list[str] = []

        def snapshot_vars(frame_locals: Dict[str, Any]) -> Dict[str, Any]:
            # Merge globals + locals, locals override
            raw: Dict[str, Any] = {}
            raw.update(global_ns)
            raw.update(frame_locals)

            snap: Dict[str, Any] = {}
            for k, v in raw.items():
                if k.startswith("__"):
                    continue
                try:
                    json.dumps(v)  # test JSON serializability
                    snap[k] = v
                except TypeError:
                    snap[k] = repr(v)
            return snap

        def tracer(frame, event, arg):
            # Only trace lines from the user's code string
            if frame.f_code.co_filename != "<user_code>":
                return tracer

            if event == "line":
                lineno = frame.f_lineno
                code_line = (
                    code_lines[lineno - 1] if 1 <= lineno <= len(code_lines) else ""
                )

                # Update outputs_so_far from captured stdout
                stdout_text = buf.getvalue()
                all_out_lines = stdout_text.splitlines()
                new_lines = all_out_lines[len(outputs_so_far):]
                outputs_so_far.extend(new_lines)

                vars_snapshot = snapshot_vars(frame.f_locals)

                steps.append({
                    "lineNo": lineno,
                    "codeLine": code_line,
                    "desc": f"Executing line {lineno}: {code_line.strip()}",
                    "varsSnapshot": vars_snapshot,
                    "outputs": list(outputs_so_far),
                    "visualizations": []   # you can add custom visualizations later
                })

            return tracer

        # Run code with stdout capture + tracer
        with contextlib.redirect_stdout(buf):
            sys.settrace(tracer)
            try:
                exec(code_obj, global_ns, local_ns)
            finally:
                sys.settrace(None)

        # If for some reason we got no steps (e.g., empty code),
        # still return something.
        if not steps:
            stdout_text = buf.getvalue()
            outputs_so_far[:] = stdout_text.splitlines()
            steps.append({
                "lineNo": 0,
                "codeLine": "",
                "desc": "Execution finished",
                "varsSnapshot": snapshot_vars({}),
                "outputs": list(outputs_so_far),
                "visualizations": []
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
