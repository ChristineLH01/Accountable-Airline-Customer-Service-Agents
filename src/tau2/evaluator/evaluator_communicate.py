"""Required-information evaluation with the pinned semantic judge or substrings."""

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from tau2.data_model.semantic_communication import (
    CommunicateScorer,
    CommunicateScoring,
    SemanticCommunicateReply,
)
from tau2.data_model.simulation import CommunicateCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

# Calibrated semantic prompts; revision is recorded by SemanticCommunicateSettings.
COMMUNICATE_JUDGE_SYSTEM_PROMPT = """
        TASK
        - You will be given a piece of expected information and a conversation between an agent and a customer.
        - Your job is to decide whether the agent communicated the expected information to the customer at any point in the conversation.
        - The information counts as communicated if its meaning was clearly conveyed to the customer. Exact wording, formatting, number/date formatting, language, or script do NOT need to match.
        - Paraphrases, translations, transliterations/romanizations, and code-switched phrasings of the expected information all count, as long as the meaning is preserved.
        - If the agent never conveyed the information, or conveyed contradictory information, it does not count.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your decision
        - `met`: `true` if the agent communicated the expected information, `false` otherwise

        Example response structure:
        {
            "reasoning": "<reasoning trace>",
            "met": <true or false>
        }
        """

COMMUNICATE_JUDGE_USER_PROMPT_TEMPLATE = """
        Did the agent communicate the following information to the user?

        expected information:
        {info_str}

        Conversation transcript:
        {transcript}
        """


class CommunicateEvaluator(EvaluatorBase[Message]):
    """Check required information with the pinned semantic judge or substrings."""

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
        *,
        scoring: CommunicateScoring,
    ) -> RewardInfo:
        """Score required facts; absent criteria are neutral and make no LLM calls."""
        semantic = scoring.scorer == CommunicateScorer.SEMANTIC
        if task.evaluation_criteria is None and not semantic:
            return RewardInfo(
                reward=1.0,
                info={"notes": "No evaluation criteria"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )
        communicate_info = (
            task.evaluation_criteria.communicate_info or []
            if task.evaluation_criteria is not None
            else []
        )
        if not communicate_info and not semantic:
            return RewardInfo(
                reward=1.0,
                info={"note": "No communicate_info to evaluate"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )
        checks = cls.evaluate_communicate_info(
            full_trajectory,
            communicate_info,
            scoring=scoring,
        )
        reward = float(all(check.met for check in checks))
        return RewardInfo(
            reward=reward,
            communicate_checks=checks,
            reward_breakdown={RewardType.COMMUNICATE: reward},
            info={
                "communicate_scoring": scoring.model_dump(mode="json"),
            }
            if semantic
            else None,
        )

    @classmethod
    def evaluate_communicate_info(
        cls,
        full_trajectory: list[Message],
        communicate_info: list[str],
        *,
        scoring: CommunicateScoring,
    ) -> list[CommunicateCheck]:
        """Check each fact; semantic mode makes one validated LLM call per fact."""
        if not communicate_info:
            return []
        judge = scoring.judge
        if scoring.scorer == CommunicateScorer.SEMANTIC:
            transcript = "\n".join(
                f"{message.role}: {message.content}"
                for message in full_trajectory
                if isinstance(message, (AssistantMessage, UserMessage))
                and message.has_text_content()
            )
            system = COMMUNICATE_JUDGE_SYSTEM_PROMPT
            checks = []
            for value in communicate_info:
                response = generate(
                    model=judge.model,
                    messages=[
                        SystemMessage(role="system", content=system),
                        UserMessage(
                            role="user",
                            content=COMMUNICATE_JUDGE_USER_PROMPT_TEMPLATE.format(
                                info_str=value, transcript=transcript
                            ),
                        ),
                    ],
                    call_name="communicate_info_eval",
                    **judge.model_args,
                )
                reply = SemanticCommunicateReply.model_validate_json(
                    extract_json_from_llm_response(response.content or "")
                )
                checks.append(
                    CommunicateCheck(
                        info=value,
                        met=reply.met,
                        justification=f"[llm_judge] {reply.reasoning}",
                    )
                )
            return checks

        outputs = []
        for info_str in communicate_info:
            found = False
            for message in full_trajectory:
                if not isinstance(message, AssistantMessage):
                    continue
                if not message.has_text_content():
                    continue
                if info_str.lower() in message.content.lower().replace(
                    ",", ""
                ):  # TODO: This could be improved!
                    found = True
                    break
            if found:
                met = True
                justification = f"Information '{info_str}' communicated in the message:\n '{message.content}'"
            else:
                met = False
                justification = f"Information '{info_str}' not communicated."
            outputs.append(
                CommunicateCheck(
                    info=info_str,
                    met=met,
                    justification=justification,
                )
            )
        return outputs
