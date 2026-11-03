from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.data_model.tasks import Action, Task
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


LLMAgentStateType = TypeVar("LLMAgentStateType", bound="LLMAgentState")


class LLMAgent(
    LLMConfigMixin, HalfDuplexAgent[LLMAgentStateType], Generic[LLMAgentStateType]
):
    """
    A half-duplex LLM agent for turn-based conversations.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> tuple[AssistantMessage, LLMAgentStateType]:
        """
        Respond to a user or tool message.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> AssistantMessage:
        """
        Generate the next message from a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        return assistant_message


AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_GT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<resolution_steps>
{resolution_steps}
</resolution_steps>
""".strip()


class LLMGTAgent(
    LLMConfigMixin, HalfDuplexAgent[LLMAgentStateType], Generic[LLMAgentStateType]
):
    """
    A GroundTruth agent that can be used to solve a task.
    This agent will receive the expected actions.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """
        Initialize the LLMAgent.
        If provide_function_args is True, the resolution steps will include the function arguments.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.provide_function_args = provide_function_args

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Only the tasks that require at least one action are valid.
        """
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> tuple[AssistantMessage, LLMAgentStateType]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_gt_response",
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    def make_agent_instructions_from_actions(self) -> str:
        """
        Make agent instructions from a list of actions
        """
        lines = []
        for i, action in enumerate(self.task.evaluation_criteria.actions):
            lines.append(
                f"[Step {i + 1}] {self.make_agent_instructions_from_action(action=action, include_function_args=self.provide_function_args)}"
            )
        return "\n".join(lines)

    @classmethod
    def make_agent_instructions_from_action(
        cls, action: Action, include_function_args: bool = False
    ) -> str:
        """
        Make agent instructions from an action.
        If the action is a user action, returns instructions for the agent to give to the user.
        If the action is an agent action, returns instructions for the agent to perform the action.
        """
        if action.requestor == "user":
            if include_function_args:
                return f"Instruct the user to perform the following action: {action.get_func_format()}."
            else:
                return f"User action: {action.name}."
        elif action.requestor == "assistant":
            if include_function_args:
                return f"Perform the following action: {action.get_func_format()}."
            else:
                return f"Assistant action: {action.name}."
        else:
            raise ValueError(f"Unknown action requestor: {action.requestor}")


# =============================================================================
# AGENT FACTORY FUNCTIONS
# =============================================================================


def create_llm_agent(tools, domain_policy, **kwargs):
    """Factory function for LLMAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def create_llm_gt_agent(tools, domain_policy, **kwargs):
    """Factory function for LLMGTAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
            - task (Task): The task to solve (required for GT agent).
    """
    return LLMGTAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
