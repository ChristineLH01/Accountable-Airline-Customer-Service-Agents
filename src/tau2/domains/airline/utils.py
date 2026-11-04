from tau2.utils.utils import domain_data_dir

AIRLINE_DATA_DIR = domain_data_dir("airline")  # data/tau2-airline
AIRLINE_DB_PATH = AIRLINE_DATA_DIR / "db.json"
AIRLINE_POLICY_PATH = AIRLINE_DATA_DIR / "policy.md"
AIRLINE_TASK_SET_PATH = AIRLINE_DATA_DIR / "tasks.json"
# Answer key for the agent's decision explanations: the final golden_decision
# and golden_reason per task (plus the annotators' labels until consensus).
AIRLINE_MANUAL_LABELS_PATH = AIRLINE_DATA_DIR / "mannual_label.json"
