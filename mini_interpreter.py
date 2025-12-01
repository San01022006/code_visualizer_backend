# mini_interpreter.py
from dataclasses import dataclass
from typing import List, Dict, Any
import re

@dataclass
class StepEvent:
    lineNo: int
    codeLine: str
    desc: str
    varsSnapshot: Dict[str, str]
    outputs: List[str]
    visualizations: List[str]

class ExprEvaluator:
    def __init__(self, vars_map: Dict[str, Any]):
        self.vars = vars_map

    def evalExpr(self, expr: str):
        s = expr.strip()
        if s == "":
            return None

        # Support simple f-strings: f"...{expr}..."
        if (s.startswith('f"') and s.endswith('"')) or (s.startswith("f'") and s.endswith("'")):
            body = s[2:-1]
            return self.eval_fstring(body)

        # method calls like obj.method(...)
        if '.' in s:
            parts = s.split('.')
            if len(parts) == 2:
                objName = parts[0].strip()
                methodCall = parts[1].strip()
                if '(' in methodCall and methodCall.endswith(')'):
                    methodName = methodCall.split('(')[0]
                    args = methodCall[methodCall.find('(')+1:-1]
                    obj = self.vars.get(objName)
                    return self.handleMethodCall(obj, methodName, args, objName)

        # function-style calls
        if '(' in s and s.endswith(')') and not s.startswith('['):
            funcName = s.split('(')[0]
            args = s[s.find('(')+1:-1]
            return self.handleFunctionCall(funcName, args)

        # list literal
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]
            if inner.strip() == "":
                return []
            parts = self.splitTopLevel(inner, ',')
            return [self.evalExpr(p.strip()) for p in parts]

        # indexing: a[b]
        idxMatch = re.match(r'^(.+)\[(.+)\]$', s)
        if idxMatch:
            arrExpr = idxMatch.group(1).strip()
            idxExpr = idxMatch.group(2).strip()
            arrVal = self.evalExpr(arrExpr)
            idxVal = self.evalExpr(idxExpr)
            if isinstance(arrVal, list) and isinstance(idxVal, (int, float)):
                idx = int(idxVal)
                if 0 <= idx < len(arrVal):
                    return arrVal[idx]
                return None

        # comparisons (>=, <=, >, <) - simple numeric compares
        for op in ['>=','<=','>','<']:
            if op in s:
                parts = s.split(op)
                if len(parts) == 2:
                    left = self.evalExpr(parts[0].strip())
                    right = self.evalExpr(parts[1].strip())
                    try:
                        if left is None or right is None:
                            return False
                        return int(left) >= int(right) if op == '>=' else \
                               int(left) <= int(right) if op == '<=' else \
                               int(left) > int(right) if op == '>' else \
                               int(left) < int(right)
                    except:
                        return False

        # integers
        if re.match(r"^-?\d+$", s):
            return int(s)

        # quoted strings (normal string literal)
        if re.match(r'^".*"$', s) or re.match(r"^'.*'$", s):
            return s[1:-1]

        # variable lookup
        if s in self.vars:
            return self.vars[s]

        # fallback: arithmetic expression
        return self.evalArithmetic(s)

    def eval_fstring(self, body: str):
        # Replace {...} expressions inside the f-string by evaluated values
        def repl(match):
            expr = match.group(1)
            val = self.evalExpr(expr)
            # For f-strings, convert lists and others to readable repr
            if isinstance(val, str):
                return val
            return self.repr_value(val)
        return re.sub(r'\{([^}]+)\}', repl, body)

    def handleMethodCall(self, obj, methodName: str, args: str, objName: str):
        if isinstance(obj, list):
            lst = obj
            if methodName in ("append", "push"):
                val = self.evalExpr(args) if args.strip() else None
                lst.append(val)
                return val
            if methodName == "pop":
                return lst.pop() if lst else None
            if methodName in ("peek","top"):
                return lst[-1] if lst else None
            if methodName == "dequeue":
                return lst.pop(0) if lst else None
            if methodName == "enqueue":
                val = self.evalExpr(args) if args.strip() else None
                lst.append(val)
                return val
            if methodName == "isEmpty":
                return len(lst) == 0
            if methodName == "size":
                return len(lst)
        return None

    def handleFunctionCall(self, funcName: str, args: str):
        fn = funcName.strip()
        if fn == "len":
            obj = self.evalExpr(args)
            if isinstance(obj, list): return len(obj)
            if isinstance(obj, str): return len(obj)
            return 0
        if fn == "range":
            # mimic Kotlin-ish 'range(n)' used in your code: returns count
            a = self.evalExpr(args)
            if isinstance(a, (int, float)): return int(a)
            return 0
        if fn == "min":
            if ',' in args:
                parts = self.splitTopLevel(args, ',')
                nums = [self.evalExpr(p.strip()) for p in parts]
                nums = [n for n in nums if isinstance(n, (int, float))]
                return min(nums) if nums else None
            else:
                arr = self.evalExpr(args)
                if isinstance(arr, list):
                    nums = [n for n in arr if isinstance(n, (int, float))]
                    return min(nums) if nums else None
        if fn == "max":
            if ',' in args:
                parts = self.splitTopLevel(args, ',')
                nums = [self.evalExpr(p.strip()) for p in parts]
                nums = [n for n in nums if isinstance(n, (int, float))]
                return max(nums) if nums else None
            else:
                arr = self.evalExpr(args)
                if isinstance(arr, list):
                    nums = [n for n in arr if isinstance(n, (int, float))]
                    return max(nums) if nums else None
        return None

    def splitTopLevel(self, s: str, delim: str):
        res = []
        depth = 0
        cur = []
        for ch in s:
            if ch in '[(':
                depth += 1
            elif ch in '])':
                depth -= 1
            if ch == delim and depth == 0:
                res.append(''.join(cur))
                cur = []
            else:
                cur.append(ch)
        if cur:
            res.append(''.join(cur))
        return res

    def evalArithmetic(self, expr: str):
        tokens = self.tokenize(expr)
        if not tokens: return None
        if len(tokens) == 1:
            return self.evalExpr(tokens[0])
        out = []
        ops = []
        prec = {"+":1,"-":1,"*":2,"/":2,"%":2}
        for t in tokens:
            if re.match(r"^-?\d+$", t) or t in self.vars:
                out.append(t)
            elif t == "(":
                ops.append(t)
            elif t == ")":
                while ops and ops[-1] != "(":
                    out.append(ops.pop())
                if ops and ops[-1] == "(":
                    ops.pop()
            elif t in prec:
                while ops and ops[-1] != "(" and prec.get(ops[-1],0) >= prec[t]:
                    out.append(ops.pop())
                ops.append(t)
            else:
                out.append(t)
        while ops:
            out.append(ops.pop())
        st = []
        for tok in out:
            if re.match(r"^-?\d+$", tok):
                st.append(int(tok))
            elif tok in self.vars:
                st.append(self.vars[tok])
            elif tok in {"+","-","*","/","%"}:
                if len(st) >= 2:
                    b = st.pop()
                    a = st.pop()
                    st.append(self.arithmeticOp(a,b,tok))
            else:
                st.append(None)
        return st[-1] if st else None

    def arithmeticOp(self, a, b, op):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        if op == "+": return int(a) + int(b)
        if op == "-": return int(a) - int(b)
        if op == "*": return int(a) * int(b)
        if op == "/": return None if int(b) == 0 else int(a) // int(b)
        if op == "%": return None if int(b) == 0 else int(a) % int(b)
        return None

    def tokenize(self, s: str):
        res = []
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c.isspace():
                i += 1
                continue
            if c.isdigit() or (c == '-' and i+1 < n and s[i+1].isdigit()):
                sb = [c]; i += 1
                while i < n and s[i].isdigit():
                    sb.append(s[i]); i += 1
                res.append(''.join(sb))
            elif c.isalpha() or c == '_':
                sb = [c]; i += 1
                while i < n and (s[i].isalnum() or s[i] == '_'):
                    sb.append(s[i]); i += 1
                res.append(''.join(sb))
            elif c in "+-*/%()[]":
                res.append(c); i += 1
            else:
                i += 1
        return res

    def repr_value(self, v):
        # helper used for f-string substitution and repr of complex objects
        if v is None:
            return "null"
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return "[" + ", ".join(self.repr_value(x) for x in v) + "]"
        return str(v)

class MiniInterpreter:
    def __init__(self, rawCode: str):
        self.rawCode = rawCode
        self.lines = rawCode.splitlines()
        self.pc = 0
        self.steps: List[StepEvent] = []
        self.outputs: List[str] = []
        self.vars: Dict[str, Any] = {}

    # ---------------- small helper to strip inline comments ----------------
    def strip_comments(self, s: str) -> str:
        """
        Remove '#' comments from s, but preserve '#' inside quoted strings.
        """
        res = []
        in_single = False
        in_double = False
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "'" and not in_double:
                in_single = not in_single
                res.append(ch)
            elif ch == '"' and not in_single:
                in_double = not in_double
                res.append(ch)
            elif ch == '#' and not in_single and not in_double:
                # comment start -> stop processing rest
                break
            else:
                res.append(ch)
            i += 1
        return ''.join(res).rstrip()

    def runAll(self):
        self.steps.clear()
        self.outputs.clear()
        self.vars.clear()
        self.pc = 0
        n = len(self.lines)
        while self.pc < n:
            if 0 <= self.pc < n:
                self.executeLine(self.pc)
            self.pc += 1

    def executeLine(self, index: int):
        if index >= len(self.lines): return
        raw = self.lines[index]
        trimmed = raw.strip()
        if not trimmed or trimmed.startswith("#"):
            return
        # micro-step: show next line to execute (helps match Python Tutor)
        self.recordStep(index, trimmed, f"Next -> {trimmed}")
        if trimmed.startswith("print"):
            inside = self.extractFunctionArgs(trimmed, "print")
            evaluator = ExprEvaluator(self.vars)
            parts = evaluator.splitTopLevel(inside, ',') if inside.strip() else []
            out_parts = []
            for p in parts:
                val = evaluator.evalExpr(p.strip())
                # For print, do not add extra quotes around strings
                if isinstance(val, str):
                    out_parts.append(val)
                else:
                    # use evaluator.repr_value for readable lists/numbers
                    out_parts.append(evaluator.repr_value(val))
            out_str = " ".join(out_parts)
            self.outputs.append(out_str)
            self.recordStep(index, trimmed, f"Output -> {out_str}")
            return
        if trimmed.startswith("for ") and " in range" in trimmed:
            self.handleForLoop(index, trimmed); return
        if trimmed.startswith("while ") and trimmed.endswith(":"):
            self.handleWhileLoop(index, trimmed); return
        if trimmed.startswith("if ") and trimmed.endswith(":"):
            self.handleIfStatement(index, trimmed); return
        if trimmed == "else:":
            self.recordStep(index, trimmed, "Else block"); return
        if "=" in trimmed and not self.containsComparison(trimmed):
            self.handleAssignment(index, trimmed); return
        if ".swap(" in trimmed:
            self.handleSwap(index, trimmed); return
        try:
            ExprEvaluator(self.vars).evalExpr(trimmed)
            self.recordStep(index, trimmed, f"Executed: {trimmed}")
        except Exception:
            self.recordStep(index, trimmed, f"Unable to execute: {trimmed}")

    def handleSwap(self, index, trimmed):
        parts = trimmed.split(".swap(")
        if len(parts) == 2:
            arrName = parts[0].strip()
            args = parts[1].rstrip(")").strip()
            argParts = args.split(",")
            if len(argParts) == 2:
                i = ExprEvaluator(self.vars).evalExpr(argParts[0].strip())
                j = ExprEvaluator(self.vars).evalExpr(argParts[1].strip())
                arr = self.vars.get(arrName)
                if isinstance(arr, list) and isinstance(i, (int,float)) and isinstance(j, (int,float)):
                    ii = int(i); jj = int(j)
                    if 0 <= ii < len(arr) and 0 <= jj < len(arr):
                        temp = arr[ii]
                        arr[ii] = arr[jj]
                        arr[jj] = temp
                        vis = self.createArrayVisualization(arr, {ii, jj})
                        self.recordStep(index, trimmed, f"Swapped {arrName}[{ii}] ↔ {arrName}[{jj}]", [vis])

    # ----------------- CORRECTLY INDENTED handleAssignment -----------------
    def handleAssignment(self, index, trimmed):
        eq = trimmed.find('=')
        left = trimmed[:eq].strip()
        right = trimmed[eq+1:].strip()

        # Remove inline comments from the right-hand side before evaluating.
        right = self.strip_comments(right)

        if '[' in left and ']' in left:
            m = re.match(r'^(.+)\[(.+)\]$', left)
            if m:
                arrName = m.group(1).strip()
                idxExpr = m.group(2).strip()
                arr = self.vars.get(arrName)
                idxVal = ExprEvaluator(self.vars).evalExpr(idxExpr)
                value = ExprEvaluator(self.vars).evalExpr(right)
                if isinstance(arr, list) and isinstance(idxVal, (int,float)):
                    idx = int(idxVal)
                    if 0 <= idx < len(arr):
                        arr[idx] = value
                        vis = self.createArrayVisualization(arr, {idx})
                        self.recordStep(index, trimmed, f"Set {arrName}[{idx}] = {self.stringify(value)}", [vis])
                        return
        value = ExprEvaluator(self.vars).evalExpr(right)
        self.vars[left] = value
        visualizations = []
        if isinstance(value, list):
            visualizations.append(self.createArrayVisualization(value))
            if self.isStack(left):
                visualizations.append(self.createStackVisualization(value))
            elif self.isQueue(left):
                visualizations.append(self.createQueueVisualization(value))
        self.recordStep(index, trimmed, f"Assign: {left} = {self.stringify(value)}", visualizations)


    def handleForLoop(self, index, trimmed):
        varName = trimmed.split("for ")[1].split(" in range")[0].strip()
        rangeExpr = trimmed.split("in range",1)[1].strip()
        rangeExpr = rangeExpr.lstrip("(").rstrip("):")
        r = ExprEvaluator(self.vars).evalExpr(rangeExpr)
        count = int(r) if isinstance(r, (int,float)) else 0
        self.recordStep(index, trimmed, f"For loop: {varName} from 0 to {count-1}")
        blockLines = self.collectBlock(index)
        for it in range(count):
            self.vars[varName] = it
            self.recordStep(index, trimmed, f"Loop iteration {it}: {varName} = {it}")
            for bl in blockLines:
                # micro-step making inner-line explicit (closer to Python Tutor)
                self.recordStep(bl, self.lines[bl].strip(), f"About to execute: {self.lines[bl].strip()}")
                self.executeLine(bl)
        self.pc = blockLines[-1] if blockLines else index

    def handleWhileLoop(self, index, trimmed):
        condExpr = trimmed[len("while"):].rstrip(":").strip()
        blockLines = self.collectBlock(index)
        iterations = 0
        maxIterations = 1000
        while iterations < maxIterations:
            condVal = ExprEvaluator(self.vars).evalExpr(condExpr)
            truth = self.isTruthy(condVal)
            self.recordStep(index, trimmed, f"While condition ({condExpr}) = {truth}")
            if not truth:
                break
            for bl in blockLines:
                self.executeLine(bl)
            iterations += 1
        if iterations >= maxIterations:
            self.recordStep(index, trimmed, "Loop terminated (max iterations reached)")
        self.pc = blockLines[-1] if blockLines else index

    def handleIfStatement(self, index, trimmed):
        condExpr = trimmed[len("if"):].rstrip(":").strip()
        truth = self.isTruthy(ExprEvaluator(self.vars).evalExpr(condExpr))
        self.recordStep(index, trimmed, f"If condition ({condExpr}) = {truth}")
        ifBlock = self.collectBlock(index)
        elseBlock = self.collectElseBlock(index)
        if truth:
            for bl in ifBlock:
                self.executeLine(bl)
        elif elseBlock is not None:
            for bl in elseBlock:
                self.executeLine(bl)
        if truth and ifBlock:
            self.pc = ifBlock[-1]
        elif not truth and elseBlock and elseBlock:
            self.pc = elseBlock[-1]
        elif ifBlock:
            self.pc = ifBlock[-1]
        else:
            self.pc = index

    def createArrayVisualization(self, lst, highlights=set()):
        sb = []
        for i,v in enumerate(lst):
            s = self.stringify(v)
            if i in highlights:
                sb.append(f"*{s}*")
            else:
                sb.append(s)
        return "Array: [" + ", ".join(sb) + "]"

    def createStackVisualization(self, lst):
        if not lst:
            return "Stack (top to bottom):\n  (empty)"
        lines = []
        for i in range(len(lst)-1, -1, -1):
            prefix = "→ " if i == len(lst)-1 else "  "
            lines.append(f"  {prefix}{self.stringify(lst[i])}")
        return "Stack (top to bottom):\n" + "\n".join(lines)

    def createQueueVisualization(self, lst):
        if not lst:
            return "Queue (front to rear):\n  Front → (empty) ← Rear"
        body = " → ".join(self.stringify(v) for v in lst)
        return "Queue (front to rear):\n  Front → " + body + " ← Rear"

    def isStack(self, varName: str): return "stack" in varName.lower()
    def isQueue(self, varName: str): return "queue" in varName.lower()

    def extractFunctionArgs(self, statement: str, funcName: str):
        start = statement.find('(', len(funcName))
        end = statement.rfind(')')
        if start != -1 and end != -1 and end > start:
            return statement[start+1:end]
        return ""

    def containsComparison(self, s: str):
        return any(x in s for x in ["==","!=","<=",">=","<",">"])

    def collectBlock(self, lineIndex: int):
        result = []
        baseIndent = self.indentOf(self.lines[lineIndex])
        i = lineIndex + 1
        while i < len(self.lines):
            line = self.lines[i]
            if line.strip() == "":
                i += 1; continue
            indent = self.indentOf(line)
            if indent <= baseIndent:
                break
            result.append(i)
            i += 1
        return result

    def collectElseBlock(self, lineIndex: int):
        ifBlock = self.collectBlock(lineIndex)
        nextIndex = (lineIndex + 1) if not ifBlock else (ifBlock[-1] + 1)
        if nextIndex >= len(self.lines):
            return None
        nextLine = self.lines[nextIndex]
        if nextLine.strip() == "else:" and self.indentOf(nextLine) == self.indentOf(self.lines[lineIndex]):
            return self.collectBlock(nextIndex)
        return None

    def indentOf(self, s: str):
        i = 0
        while i < len(s) and s[i] == ' ':
            i += 1
        return i

    def isTruthy(self, v):
        if v is None: return False
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)): return int(v) != 0
        if isinstance(v, str): return len(v) > 0
        if isinstance(v, list): return len(v) > 0
        return True

    def stringify(self, v):
        # Keep variable snapshots similar to Kotlin original:
        # strings are quoted, lists are shown as [a, b, c], None -> null
        if v is None: return "null"
        if isinstance(v, str): return f"\"{v}\""
        if isinstance(v, list):
            inner = ", ".join(self.stringify(x) for x in v)
            return "[" + inner + "]"
        return str(v)

    def recordStep(self, lineIndex, codeLine, desc, visualizations=None):
        snap = {}
        for k,v in self.vars.items():
            snap[k] = self.stringify(v)
        visuals = visualizations or []
        se = StepEvent(lineNo=lineIndex+1, codeLine=codeLine.strip(), desc=desc, varsSnapshot=snap, outputs=list(self.outputs), visualizations=visuals)
        self.steps.append(se)
