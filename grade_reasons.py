"""Grade the reasons agents gave the customer, on top of the decision score.

run_midway_reason.sh calls this after the midway runs. For every simulation,
one LLM judge call checks what the agent told the customer against a rubric
written from the golden labels (data/tau2-airline/reason_rubric.json, built
from golden_decision and golden_reason in mannual_label.json). Each run then
gets two scores, each with pass^k:

  decision           DB x COMMUNICATE, tau2's own reward (as in run_midway.sh)
  decision x reason  DB x COMMUNICATE x REASON

REASON is 1 when, for every request in the task's rubric, the agent told the
customer the rubric's outcome, conveyed every required reason element, did
nothing on the request's must_not list, and said nothing that contradicts the
golden facts or the policy. The judge returns those verdicts; the rule that
combines them is fixed here in code (`score`).

The judge sees the policy, the golden label, the rubric and only the messages
the customer saw: no tool calls or tool results, so Method B's internal
checklist is invisible to it. The judge model, prompt and rubric are the same
for every run, so methods A and B are graded the same way.

The judge's raw verdicts are cached in data/simulations/<run>/reason_grades.json
and reused until anything in the judge's input or settings changes (model,
prompt, rubric entry, label, policy or transcript). The score is recomputed
from the verdicts on every call, so changing `score` needs no new judge calls.

Usage, from the repo root:
    uv run python grade_reasons.py midway_reason_a midway_reason_b
    uv run python grade_reasons.py passk_a passk_b,passk_b_retry  # a comma joins runs
    uv run python grade_reasons.py midway_a midway_b --regrade    # ignore the cache

Options: --judge-model, --judge-args JSON, --max-concurrency N, --regrade.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from summarize_runs import SIM_ROOT, pass_hat_k
from tau2.config import DEFAULT_LLM_REASON_JUDGE, DEFAULT_LLM_REASON_JUDGE_ARGS
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.data_model.simulation import Results
from tau2.domains.airline.golden_reasons import load_golden_reasons
from tau2.domains.airline.utils import AIRLINE_POLICY_PATH
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

RUBRIC_PATH = (
    Path(__file__).resolve().parent / "data" / "tau2-airline" / "reason_rubric.json"
)
CACHE_FILE = "reason_grades.json"
PROMPT_VERSION = "reason-rubric-judge-v1"
JUDGE_CALL_NAME = "reason_rubric_judge"
JUDGE_VALIDATION_RETRIES = 2

SYSTEM_PROMPT = """\
You check one thing about an airline customer-service conversation: whether the
agent told the customer the correct decision on each request and the correct
policy-based reason for it. You grade against a rubric written from the golden
(correct) decision and reason.

You are given:
- POLICY: the airline policy the agent must follow.
- GOLDEN DECISION and GOLDEN REASON: the correct decision on the customer's
  requests and the rules and facts behind it. Treat their facts as true.
- RUBRIC: for each customer request, the outcome to look for, the reason
  elements to look for, and things the agent must not do.
- TRANSCRIPT: the messages between the customer and the agent, as the customer
  saw them. You do not see the agent's tool calls or their results. A customer
  message such as ###STOP### or ###OUT-OF-SCOPE### means the customer ended the
  conversation.

Grade only what the agent said to the customer. Fill in the JSON TEMPLATE,
keeping every id and replacing every <...> placeholder. Every quote must be
copied exactly from one place in the agent's messages: no ellipses, no joining
of separate passages, no rewording. If the words are spread over several
places, quote the clearest one.

For each rubric request:

1. outcome
   - "told": the agent clearly told the customer the rubric outcome, or one of
     its also_accepted outcomes. If the agent changed its answer, grade the
     last answer it gave. A transfer to a human agent after a clear outcome
     does not undo the outcome, even when the agent says a human will look at
     the customer's dispute; only an explicit reversal does.
   - "not_told": the agent told the customer a different outcome, left the
     request undecided, or never gave an outcome, for example because the
     conversation ended first.
   - "withdrawn": the agent never told the customer an outcome for this
     request because the customer explicitly dropped it first, and not because
     the agent stalled or failed to answer. This is rare. If the agent did tell
     the customer an outcome, use "told" or "not_told" even if the customer had
     said they would accept a refusal.
   outcome_quote: the agent's words that give the outcome, or empty.

2. elements, for each one:
   - applies: false only when the element has a "when" condition and that did
     not happen in the conversation. Otherwise true.
   - conveyed: true when the agent's messages state this fact or rule to the
     customer, in any wording, so that the customer would learn it. Paraphrase
     is fine; dates, amounts and flight numbers are not needed unless the
     element asks for them. Naming the cabin (basic economy or economy) conveys
     that it is not business. It is not conveyed if the agent only asked about
     it, only hinted at it, said it about another reservation or request, or
     said something that contradicts it.
   - quote: the agent's words that convey it; empty when not conveyed.

3. must_not, for each item: violated is true when the agent did it; quote gives
   the agent's words, or empty.

4. note: one or two sentences on the request.

Then contradictions: one item {"quote": "<agent's words>", "why": "<what is
wrong>"} for every statement the agent made to the customer about the rubric's
requests (their outcomes, their reasons, and the facts and rules behind them)
that contradicts the golden facts or misstates the POLICY, such as a wrong
cabin, booking date, insurance status, flight status or amount, or a rule
stated differently from the POLICY. Use an empty list when there are none.
Read each statement in context: a statement about this customer's reservation
is correct if it is true for that reservation, even if it would not hold for
every reservation. Do not list:
- information the customer asked for outside the rubric's requests (for
  example a trip total); it is graded elsewhere;
- facts the golden decision and reason do not cover, unless the transcript or
  the POLICY shows they are wrong;
- saying the policy does not allow something the POLICY does not provide for;
- wording slips that do not change the meaning;
- statements the agent itself corrected later in the conversation.

Do not grade politeness, authentication, confirmation steps or anything else
outside the decisions and their reasons. Extra correct information is fine.

Return only the completed JSON object, with no text before or after it.
"""

USER_TEMPLATE = """\
POLICY:
{policy}

GOLDEN DECISION:
{golden_decision}

GOLDEN REASON:
{golden_reason}

RUBRIC:
{rubric}

TRANSCRIPT:
{transcript}

JSON TEMPLATE:
{template}
"""


# ---- Rubric ---------------------------------------------------------------


class RubricElement(BaseModel):
    """A fact or rule the agent's reason must contain."""

    model_config = ConfigDict(extra="forbid")
    id: str
    text: str
    when: Optional[str] = None  # a key of Rubric.conditions; None: always required


class RubricMustNot(BaseModel):
    """Something the agent must not do, such as accepting a false claim."""

    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class RubricRequest(BaseModel):
    """One customer request: the outcome to tell and the reason to give."""

    model_config = ConfigDict(extra="forbid")
    id: str
    request: str
    outcome: str
    also_accepted: list[str] = Field(default_factory=list)
    elements: list[RubricElement] = Field(min_length=1)
    must_not: list[RubricMustNot] = Field(default_factory=list)


class TaskRubric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[RubricRequest] = Field(min_length=1)


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    principles: list[str]
    conditions: dict[str, str]
    tasks: dict[str, TaskRubric]

    @model_validator(mode="after")
    def _check_ids(self) -> "Rubric":
        for task_id, task in self.tasks.items():
            _unique([r.id for r in task.requests], f"task {task_id}")
            for r in task.requests:
                ids = [e.id for e in r.elements] + [m.id for m in r.must_not]
                _unique(ids, f"task {task_id} request {r.id}")
                for e in r.elements:
                    if e.when is not None and e.when not in self.conditions:
                        raise ValueError(
                            f"task {task_id} {r.id}.{e.id}: unknown condition {e.when!r}"
                        )
        return self


def _unique(ids: list[str], where: str) -> None:
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{where}: duplicate ids {dupes}")


def load_rubric(path: Path = RUBRIC_PATH) -> Rubric:
    return Rubric.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---- Judge reply ----------------------------------------------------------


class ElementVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    applies: bool
    conveyed: bool
    quote: str = ""


class MustNotVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    violated: bool
    quote: str = ""


class RequestVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    outcome: Literal["told", "not_told", "withdrawn"]
    outcome_quote: str = ""
    elements: list[ElementVerdict] = Field(default_factory=list)
    must_not: list[MustNotVerdict] = Field(default_factory=list)
    note: str = ""


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    quote: str
    why: str


class JudgeReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    requests: list[RequestVerdict] = Field(min_length=1)
    contradictions: list[Contradiction] = Field(default_factory=list)
    summary: str = ""


def reply_model_for(task: TaskRubric) -> type[JudgeReply]:
    """JudgeReply that must answer exactly the rubric's requests and items.

    A reply that skips or invents an id fails validation, so generate() asks
    the judge to correct it instead of the code guessing a verdict.
    """
    expected = {
        r.id: (sorted(e.id for e in r.elements), sorted(m.id for m in r.must_not))
        for r in task.requests
    }

    class RubricReply(JudgeReply):
        @model_validator(mode="after")
        def _matches_rubric(self) -> "RubricReply":
            got = sorted(r.id for r in self.requests)
            if got != sorted(expected):
                raise ValueError(f"request ids {got}, expected {sorted(expected)}")
            for r in self.requests:
                elements, must_not = expected[r.id]
                if sorted(e.id for e in r.elements) != elements:
                    raise ValueError(f"{r.id}: element ids must be {elements}")
                if sorted(m.id for m in r.must_not) != must_not:
                    raise ValueError(f"{r.id}: must_not ids must be {must_not}")
            return self

    return RubricReply


# ---- Judge input ----------------------------------------------------------


def agent_words(content: str) -> str:
    """The text of an agent message, unwrapped from a JSON object if it is one.

    The agents often reply with an object such as {"message": "..."} (sometimes
    with more string fields, all of which the customer receives). Decoding it
    turns escaped newlines and quotes back into the text the customer reads.
    """
    text = content.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return content
        if (
            isinstance(data, dict)
            and data
            and all(isinstance(v, str) for v in data.values())
        ):
            return "\n\n".join(data.values())
    return content


def customer_view(messages: list) -> tuple[str, str]:
    """The conversation as the customer saw it, and the agent's words alone."""
    lines, agent = [], []
    for m in messages:
        if isinstance(m, UserMessage) and m.has_text_content():
            lines.append(f"customer: {m.content}")
        elif isinstance(m, AssistantMessage) and m.has_text_content():
            words = agent_words(m.content)
            lines.append(f"agent: {words}")
            agent.append(words)
    return "\n\n".join(lines), "\n".join(agent)


def rubric_for_judge(task: TaskRubric, conditions: dict[str, str]) -> list[dict]:
    """The task's rubric as the judge sees it, with conditions spelled out."""
    out = []
    for r in task.requests:
        item = {"id": r.id, "request": r.request, "outcome": r.outcome}
        if r.also_accepted:
            item["also_accepted"] = r.also_accepted
        item["elements"] = [
            {
                "id": e.id,
                "text": e.text,
                **({"when": conditions[e.when]} if e.when else {}),
            }
            for e in r.elements
        ]
        item["must_not"] = [{"id": m.id, "text": m.text} for m in r.must_not]
        out.append(item)
    return out


def reply_template(task: TaskRubric) -> dict:
    either, words = "<true or false>", "<agent's words, or empty>"
    return {
        "requests": [
            {
                "id": r.id,
                "outcome": "<told, not_told or withdrawn>",
                "outcome_quote": words,
                "elements": [
                    {"id": e.id, "applies": either, "conveyed": either, "quote": words}
                    for e in r.elements
                ],
                "must_not": [
                    {"id": m.id, "violated": either, "quote": words} for m in r.must_not
                ],
                "note": "<one or two sentences>",
            }
            for r in task.requests
        ],
        "contradictions": [],
        "summary": "<one sentence>",
    }


# ---- Scoring --------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9$%]+", " ", text.lower()).split())


def quote_found(quote: str, agent_norm: str) -> Optional[bool]:
    """Whether a quote appears in the agent's messages; None for an empty quote."""
    q = _norm(quote)
    return (q in agent_norm) if q else None


def score(task: TaskRubric, reply: JudgeReply, agent_text: str) -> dict:
    """Combine the judge's verdicts into REASON (1.0 or 0.0) with the details.

    A request passes when its outcome was told, every applicable element was
    conveyed and no must_not item was violated; a request the customer withdrew
    passes if no must_not item was violated. REASON is 1.0 when every request
    passes and the judge found no contradictions. An element without a `when`
    condition always applies, whatever the judge says.
    """
    agent_norm = _norm(agent_text)
    verdicts = {r.id: r for r in reply.requests}
    requests, checks, failed = [], [], []

    def check(req_id, name, kind, applies, ok, verdict, quote):
        checks.append(
            {
                "request": req_id,
                "check": name,
                "kind": kind,
                "applies": applies,
                "ok": ok,
                "verdict": verdict,
                "quote": quote,
                "quote_found": quote_found(quote, agent_norm),
            }
        )

    for req in task.requests:
        v = verdicts[req.id]
        withdrawn = v.outcome == "withdrawn"
        element_v = {e.id: e for e in v.elements}
        must_not_v = {m.id: m for m in v.must_not}
        check(
            req.id,
            "outcome",
            "outcome",
            True,
            v.outcome != "not_told",
            v.outcome,
            v.outcome_quote,
        )

        missing = []
        for el in req.elements:
            e = element_v[el.id]
            applies = (e.applies if el.when else True) and not withdrawn
            ok = e.conveyed or not applies
            if not ok:
                missing.append(el.id)
            verdict = "conveyed" if e.conveyed else "not conveyed"
            check(req.id, el.id, "element", applies, ok, verdict, e.quote)

        violated = []
        for mn in req.must_not:
            m = must_not_v[mn.id]
            if m.violated:
                violated.append(mn.id)
            verdict = "violated" if m.violated else "kept"
            check(req.id, mn.id, "must_not", True, not m.violated, verdict, m.quote)

        passed = v.outcome != "not_told" and not missing and not violated
        if v.outcome == "not_told":
            failed.append(f"{req.id}:outcome")
        failed += [f"{req.id}:{x}" for x in missing + violated]
        requests.append(
            {
                "id": req.id,
                "outcome": v.outcome,
                "passed": passed,
                "missing": missing,
                "violated": violated,
                "note": v.note,
            }
        )

    contradictions = [
        {
            "quote": c.quote,
            "why": c.why,
            "quote_found": quote_found(c.quote, agent_norm),
        }
        for c in reply.contradictions
    ]
    if contradictions:
        failed.append(f"contradictions:{len(contradictions)}")
    met = all(r["passed"] for r in requests) and not contradictions
    unverified = sum(c["quote_found"] is False for c in checks + contradictions)
    return {
        "reason": 1.0 if met else 0.0,
        "failed": failed,
        "requests": requests,
        "checks": checks,
        "contradictions": contradictions,
        "summary": reply.summary,
        "quotes_unverified": unverified,
    }


# ---- Grader ---------------------------------------------------------------


class Grader:
    """One judge, prompt and rubric, applied the same way to every simulation."""

    def __init__(
        self, judge_model: str, judge_args: dict, rubric_path: Path = RUBRIC_PATH
    ):
        self.judge_model = judge_model
        self.judge_args = dict(judge_args)
        self.rubric = load_rubric(rubric_path)
        self.rubric_sha = hashlib.sha256(Path(rubric_path).read_bytes()).hexdigest()
        self.labels = load_golden_reasons()
        self.policy = AIRLINE_POLICY_PATH.read_text(encoding="utf-8")
        self.prompt_sha = hashlib.sha256(
            (SYSTEM_PROMPT + USER_TEMPLATE).encode()
        ).hexdigest()

    def gradable(self, task_id: str) -> bool:
        label = self.labels.get(task_id)
        return task_id in self.rubric.tasks and label is not None and label.is_labeled

    def excluded(self, task_id: str) -> bool:
        label = self.labels.get(task_id)
        return bool(label and label.excluded_from_scoring)

    def prompt(self, task_id: str, transcript: str) -> str:
        task = self.rubric.tasks[task_id]
        label = self.labels[task_id]
        return USER_TEMPLATE.format(
            policy=self.policy,
            golden_decision=label.golden_decision,
            golden_reason=label.golden_reason,
            rubric=json.dumps(rubric_for_judge(task, self.rubric.conditions), indent=2),
            transcript=transcript or "(no messages)",
            template=json.dumps(reply_template(task), indent=2),
        )

    def fingerprint(self, task_id: str, messages: list) -> str:
        """Changes whenever the judge's input or settings change."""
        transcript, _ = customer_view(messages)
        payload = [
            self.judge_model,
            self.judge_args,
            SYSTEM_PROMPT,
            self.prompt(task_id, transcript),
        ]
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def score(self, task_id: str, messages: list, verdict: dict) -> dict:
        """Score a stored judge verdict; the scoring rule is applied fresh each time."""
        task = self.rubric.tasks[task_id]
        _, agent_text = customer_view(messages)
        reply = reply_model_for(task).model_validate(verdict["reply"])
        grade = score(task, reply, agent_text)
        grade["judge_cost"] = verdict.get("judge_cost")
        return grade

    def judge(self, task_id: str, messages: list) -> dict:
        """One judge call for one simulation; returns the verdict to store."""
        task = self.rubric.tasks[task_id]
        transcript, _ = customer_view(messages)
        reply_model = reply_model_for(task)
        prompt = self.prompt(task_id, transcript)
        response = generate(
            model=self.judge_model,
            messages=[
                SystemMessage(role="system", content=SYSTEM_PROMPT),
                UserMessage(role="user", content=prompt),
            ],
            call_name=JUDGE_CALL_NAME,
            response_model=reply_model,
            response_validation_retries=JUDGE_VALIDATION_RETRIES,
            **self.judge_args,
        )
        reply = reply_model.model_validate_json(
            extract_json_from_llm_response(response.content or "")
        )
        return {
            "judge_model": self.judge_model,
            "prompt_version": PROMPT_VERSION,
            "judge_cost": getattr(response, "cost", None),
            "reply": reply.model_dump(),
        }


class Cache:
    """Judge verdicts of one run, saved next to its results."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.verdicts = {}
        if path.exists():
            self.verdicts = json.loads(path.read_text(encoding="utf-8")).get(
                "verdicts", {}
            )

    def get(self, key: str, fingerprint: str) -> Optional[dict]:
        entry = self.verdicts.get(key)
        return (
            entry["verdict"] if entry and entry["fingerprint"] == fingerprint else None
        )

    def put(self, key: str, fingerprint: str, verdict: dict) -> None:
        with self.lock:
            self.verdicts[key] = {"fingerprint": fingerprint, "verdict": verdict}
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"verdicts": self.verdicts}, indent=1), encoding="utf-8"
            )
            os.replace(tmp, self.path)


# ---- Runs -----------------------------------------------------------------


def resolve(run: str) -> Path:
    """A run name or path -> what Results.load reads (results.json or a dir run)."""
    path = Path(run)
    if not path.exists():
        path = Path(SIM_ROOT) / run
    if path.is_dir() and not (path / "simulations").is_dir():
        path = path / "results.json"
    if not path.exists():
        sys.exit(f"No run found at {path}")
    return path


def decision_of(sim) -> Optional[float]:
    """DB x COMMUNICATE: tau2's reward without a REASON gate; None if unfinished."""
    info = sim.reward_info
    if info is None:
        return None
    return info.task_reward if info.task_reward is not None else info.reward


def grade_runs(
    groups: list[str], grader: Grader, regrade: bool, workers: int
) -> list[dict]:
    """One row per simulation, grading those without a valid cached verdict."""
    rows, jobs = [], []
    for group in groups:
        parts = group.split(",")
        for part in parts:
            path = resolve(part)
            cache = Cache((path if path.is_dir() else path.parent) / CACHE_FILE)
            for sim in Results.load(path).simulations:
                task_id = str(sim.task_id)
                row = {
                    "run": parts[0],
                    "task_id": task_id,
                    "trial": sim.trial,
                    "excluded": grader.excluded(task_id),
                    "termination": sim.termination_reason.value,
                    "decision": decision_of(sim),
                    "grade": None,
                    "note": "",
                }
                rows.append(row)
                if row["decision"] is None:
                    row["note"] = "unfinished"
                elif not grader.gradable(task_id):
                    row["note"] = "no rubric or label for this task"
                else:
                    key = f"{task_id}|{sim.trial}|{sim.id}"
                    fp = grader.fingerprint(task_id, sim.messages)
                    verdict = None if regrade else cache.get(key, fp)
                    row["messages"] = sim.messages
                    row["verdict"] = verdict
                    if verdict is None:
                        jobs.append((row, cache, key, fp))

    reused = len([r for r in rows if r.get("verdict")])
    print(f"Grading {len(jobs)} simulations ({reused} verdicts reused from cache)")
    done = [0]
    done_lock = threading.Lock()

    def work(job):
        row, cache, key, fp = job
        try:
            row["verdict"] = grader.judge(row["task_id"], row["messages"])
            cache.put(key, fp, row["verdict"])
        except Exception as e:  # keep going; a rerun retries it
            row["note"] = f"judge failed: {type(e).__name__}: {e}"[:300]
        with done_lock:
            done[0] += 1
            n = done[0]
        if n % 10 == 0 or n == len(jobs):
            print(f"  graded {n}/{len(jobs)}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, jobs))

    for row in rows:
        verdict = row.pop("verdict", None)
        messages = row.pop("messages", None)
        if verdict:
            row["grade"] = grader.score(row["task_id"], messages, verdict)
        row["reason"] = row["grade"]["reason"] if row["grade"] else None
        both = row["decision"] is not None and row["reason"] is not None
        row["decision_x_reason"] = row["decision"] * row["reason"] if both else None
    order = {g.split(",")[0]: i for i, g in enumerate(groups)}
    rows.sort(key=lambda r: (order[r["run"]], int(r["task_id"]), r["trial"]))
    return rows


# ---- Report ---------------------------------------------------------------


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else ""
    if isinstance(value, float):
        return f"{value:.3f}" if value not in (0.0, 1.0) else f"{value:.0f}"
    return str(value)


def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(fmt(r.get(c)).ljust(widths[c]) for c in columns).rstrip())


def score_summary(rows: list[dict], key: str) -> Optional[dict]:
    """Average and pass^k of one score; unfinished or ungraded trials left out."""
    by_task = {}
    for r in rows:
        if r[key] is not None:
            by_task.setdefault(r["task_id"], []).append(r[key])
    if not by_task:
        return None
    n_min = min(len(v) for v in by_task.values())
    values = [x for v in by_task.values() for x in v]
    passk = []
    for k in range(1, n_min + 1):
        vals = [
            pass_hat_k(len(v), sum(x >= 1.0 for x in v), k) for v in by_task.values()
        ]
        passk.append(sum(vals) / len(vals))
    return {
        "tasks": len(by_task),
        "trials": n_min,
        "avg": sum(values) / len(values),
        "passk": passk,
    }


def run_summaries(rows: list[dict], runs: list[str]) -> list[dict]:
    out = []
    for run in runs:
        run_rows = [r for r in rows if r["run"] == run]
        groups = [("all", run_rows)]
        if any(r["excluded"] for r in run_rows):
            excluded = sorted(
                {r["task_id"] for r in run_rows if r["excluded"]}, key=int
            )
            label = f"without {', '.join(excluded)}"
            groups.append((label, [r for r in run_rows if not r["excluded"]]))
        for group, group_rows in groups:
            for key, name in (
                ("decision", "decision"),
                ("decision_x_reason", "decision x reason"),
            ):
                s = score_summary(group_rows, key)
                if s is None:
                    continue
                row = {
                    "run": run,
                    "tasks": f"{group} ({s['tasks']})",
                    "trials": s["trials"],
                    "score": name,
                    "avg": s["avg"],
                }
                row.update({f"pass^{k}": v for k, v in enumerate(s["passk"], 1)})
                out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def report(rows: list[dict], runs: list[str], names: list[str]) -> None:
    print("\nPer simulation (decision = DB x COMMUNICATE; reason = REASON)\n")
    table = [
        {
            **r,
            "failed checks": "; ".join(r["grade"]["failed"])
            if r["grade"]
            else r["note"],
        }
        for r in rows
    ]
    print_table(
        table,
        [
            "run",
            "task_id",
            "trial",
            "excluded",
            "decision",
            "reason",
            "decision_x_reason",
            "failed checks",
        ],
    )

    summaries = run_summaries(rows, runs)
    if summaries:
        print(
            "\nPer run (pass^k as in summarize_runs.py; unfinished or ungraded trials left out)\n"
        )
        k_max = max(
            int(c.split("^")[1]) for s in summaries for c in s if c.startswith("pass^")
        )
        print_table(
            summaries,
            ["run", "tasks", "trials", "score", "avg"]
            + [f"pass^{k}" for k in range(1, k_max + 1)],
        )

    print("\nFailed reason checks per run (count over all simulations)")
    for run in runs:
        counts = Counter(
            f.split(":")[1] if not f.startswith("contradictions") else "contradictions"
            for r in rows
            if r["run"] == run and r["grade"]
            for f in r["grade"]["failed"]
        )
        listed = ", ".join(f"{k} {v}" for k, v in counts.most_common()) or "none"
        print(f"  {run}: {listed}")

    ungraded = [
        r for r in rows if r["grade"] is None and r["note"].startswith("judge failed")
    ]
    if ungraded:
        print(
            f"\n{len(ungraded)} simulations could not be graded; rerun to retry them:"
        )
        for r in ungraded:
            print(f"  {r['run']} task {r['task_id']} trial {r['trial']}: {r['note']}")
    unverified = sum(r["grade"]["quotes_unverified"] for r in rows if r["grade"])
    if unverified:
        print(
            f"\nNote: {unverified} judge quotes were not found verbatim in the agent's messages (see quote_found in the checks CSV)."
        )
    cost = sum(r["grade"].get("judge_cost") or 0 for r in rows if r["grade"])
    print(f"\nJudge cost of all verdicts shown: ${cost:.4f}")

    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(names))
    write_csv(
        Path(SIM_ROOT) / f"{stem}_reason_grades.csv",
        [
            {
                "run": r["run"],
                "task_id": r["task_id"],
                "trial": r["trial"],
                "excluded": r["excluded"],
                "termination": r["termination"],
                "decision": r["decision"],
                "reason": r["reason"],
                "decision_x_reason": r["decision_x_reason"],
                "failed_checks": "; ".join(r["grade"]["failed"]) if r["grade"] else "",
                "contradictions": " | ".join(
                    c["quote"] + " -> " + c["why"] for c in r["grade"]["contradictions"]
                )
                if r["grade"]
                else "",
                "judge_summary": r["grade"]["summary"] if r["grade"] else r["note"],
            }
            for r in rows
        ],
    )
    write_csv(
        Path(SIM_ROOT) / f"{stem}_reason_checks.csv",
        [
            {"run": r["run"], "task_id": r["task_id"], "trial": r["trial"], **c}
            for r in rows
            if r["grade"]
            for c in r["grade"]["checks"]
        ],
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "runs", nargs="+", help="run names under data/simulations, or paths"
    )
    parser.add_argument("--judge-model", default=DEFAULT_LLM_REASON_JUDGE)
    parser.add_argument(
        "--judge-args",
        type=json.loads,
        default=dict(DEFAULT_LLM_REASON_JUDGE_ARGS),
        metavar="JSON",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=4, help="judge calls at once"
    )
    parser.add_argument("--regrade", action="store_true", help="ignore cached verdicts")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    grader = Grader(args.judge_model, args.judge_args)
    print("== Reason grading: the same judge, prompt and rubric for every run")
    print(f"judge   {grader.judge_model} {json.dumps(grader.judge_args)}")
    print(f"prompt  {PROMPT_VERSION} (sha {grader.prompt_sha[:8]})")
    print(
        f"rubric  {grader.rubric.version}, {len(grader.rubric.tasks)} tasks (sha {grader.rubric_sha[:8]})"
    )

    rows = grade_runs(args.runs, grader, args.regrade, args.max_concurrency)
    report(rows, [g.split(",")[0] for g in args.runs], args.runs)


if __name__ == "__main__":
    main()
