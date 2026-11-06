# =============================================================================
# SIMULATION DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "ERROR"

# =============================================================================
# LLM DEFAULTS (overridable via CLI)
# =============================================================================
# Every model is called through OpenRouter: model names are OpenRouter model IDs
# (https://openrouter.ai/models) and one OPENROUTER_API_KEY covers all of them.
# These reasoning models take `reasoning_effort` and do not accept `temperature`.
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "openai/gpt-6.1-sol"
DEFAULT_LLM_USER = "openai/gpt-6.1-sol"
# The agent under evaluation reasons at medium effort; the user simulator only
# role-plays its scenario, so low effort keeps it fast and cheap (as in the
# released tau-bench 2.0 text benchmark's user simulator).
DEFAULT_LLM_ARGS_AGENT = {"reasoning_effort": "medium"}
DEFAULT_LLM_ARGS_USER = {"reasoning_effort": "low"}

# User simulator of the released tau-bench 2.0 text benchmark (text-v1). Pass
# these with --user-llm / --user-llm-args to match its conditions.
TEXT_BENCHMARK_LLM_USER = "openai/gpt-5.5"
TEXT_BENCHMARK_LLM_ARGS_USER = {"reasoning_effort": "low"}

# LLM judges: communicate-info scoring and NL assertions (reward), and the
# conversation review / authentication classification (diagnostics only).
DEFAULT_LLM_JUDGE = "openai/gpt-6-luna"
DEFAULT_LLM_JUDGE_ARGS = {"reasoning_effort": "medium"}

DEFAULT_SEMANTIC_COMMUNICATE_MODEL = DEFAULT_LLM_JUDGE
DEFAULT_SEMANTIC_COMMUNICATE_MODEL_ARGS = dict(DEFAULT_LLM_JUDGE_ARGS)

DEFAULT_LLM_NL_ASSERTIONS = DEFAULT_LLM_JUDGE
DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(DEFAULT_LLM_JUDGE_ARGS)

# Decision-reason judge (--reason-scoring): grades the decision and the
# policy-based reason the agent told the customer against the golden labels in
# data/tau2-airline/mannual_label.json.
DEFAULT_LLM_REASON_JUDGE = DEFAULT_LLM_JUDGE
DEFAULT_LLM_REASON_JUDGE_ARGS = dict(DEFAULT_LLM_JUDGE_ARGS)

# Post-hoc conversation review (--auto-review, `tau2 review`); never affects reward.
DEFAULT_LLM_REVIEW = DEFAULT_LLM_JUDGE
DEFAULT_LLM_REVIEW_ARGS = dict(DEFAULT_LLM_JUDGE_ARGS)

# Helper agent of the interactive environment CLI (`make env-cli`).
DEFAULT_LLM_ENV_INTERFACE = "openai/gpt-6.1-sol"
DEFAULT_LLM_ENV_INTERFACE_ARGS = {"reasoning_effort": "low"}

# LLM debug logging
DEFAULT_LLM_LOG_MODE = "latest"  # Options: "all", "latest"

# =============================================================================
# LLM INFRASTRUCTURE (fixed operational constants)
# =============================================================================
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_MIN_WAIT = 1.0  # seconds

# LiteLLM cache
LLM_CACHE_ENABLED = False
DEFAULT_LLM_CACHE_TYPE = "redis"

# Redis (fixed infrastructure config)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""
REDIS_PREFIX = "tau2"
REDIS_CACHE_VERSION = "v1"
REDIS_CACHE_TTL = 60 * 60 * 24 * 30

# Langfuse
USE_LANGFUSE = False

# =============================================================================
# DISPLAY
# =============================================================================
TERM_DARK_MODE = True
