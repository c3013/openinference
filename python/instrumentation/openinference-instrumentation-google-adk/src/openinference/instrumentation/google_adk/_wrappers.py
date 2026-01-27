import base64
import inspect
import json
import logging
import time
from abc import ABC
from contextlib import ExitStack
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    OrderedDict,
    TypedDict,
    TypeVar,
)

import wrapt
from google.adk import Runner
from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Meter
from opentelemetry.trace import StatusCode, get_current_span
from opentelemetry.util.types import AttributeValue
from typing_extensions import NotRequired, ParamSpec

from openinference.instrumentation import (
    get_attributes_from_context,
    safe_json_dumps,
    using_session,
    using_user,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

P = ParamSpec("P")
T = TypeVar("T")


class _WithTracer(ABC):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        meter: Optional[Meter] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._meter = meter


class _RunnerRunAsyncKwargs(TypedDict):
    user_id: str
    session_id: str
    new_message: types.Content
    run_config: NotRequired[RunConfig]


class _RunnerRunAsync(_WithTracer):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        meter: Optional[Meter] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(tracer, meter, *args, **kwargs)
        self._workflow_duration_histogram = None
        if self._meter:
            self._workflow_duration_histogram = self._meter.create_histogram(
                name="gen_ai.workflow.duration",
                description="Duration of workflow operations",
                unit="s",
            )

    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Event, None]],
        instance: Runner,
        args: tuple[Any, ...],
        kwargs: _RunnerRunAsyncKwargs,
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        meter = self._meter
        workflow_duration_histogram = self._workflow_duration_histogram if self._meter else None
        workflow_name = instance.app_name
        name = f"invocation [{workflow_name}]"
        attributes = dict(get_attributes_from_context())
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.CHAIN.value

        # Add gen_ai.workflow.name attribute
        attributes["gen_ai.workflow.name"] = workflow_name

        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        try:
            attributes[SpanAttributes.INPUT_VALUE] = json.dumps(
                arguments,
                default=_default,
                ensure_ascii=False,
            )
            attributes[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
        except Exception:
            logger.exception(f"Failed to get attribute: {SpanAttributes.INPUT_VALUE}.")

        # Add gen_ai.input.messages from new_message
        if new_message := kwargs.get("new_message"):
            try:
                if hasattr(new_message, "parts") and new_message.parts:
                    for i, part in enumerate(new_message.parts):
                        if hasattr(part, "text") and part.text:
                            attributes[f"gen_ai.input.messages.0.content.{i}.text"] = part.text
            except Exception:
                logger.exception("Failed to get gen_ai.input.messages attribute.")

        if (user_id := kwargs.get("user_id")) is not None:
            attributes[SpanAttributes.USER_ID] = user_id
        if (session_id := kwargs.get("session_id")) is not None:
            attributes[SpanAttributes.SESSION_ID] = session_id

        class _AsyncGenerator(wrapt.ObjectProxy):  # type: ignore[misc]
            __wrapped__: AsyncGenerator[Event, None]

            async def __aiter__(self) -> Any:
                start_time = time.time()
                error_type = None
                with ExitStack() as stack:
                    span = stack.enter_context(
                        tracer.start_as_current_span(
                            name=name,
                            attributes=attributes,
                        )
                    )
                    if user_id is not None:
                        stack.enter_context(using_user(user_id))
                    if session_id is not None:
                        stack.enter_context(using_session(session_id))
                    try:
                        async for event in self.__wrapped__:
                            if event.is_final_response():
                                try:
                                    output_value = event.model_dump_json(exclude_none=True)
                                    span.set_attribute(
                                        SpanAttributes.OUTPUT_VALUE,
                                        output_value,
                                    )
                                    span.set_attribute(
                                        SpanAttributes.OUTPUT_MIME_TYPE,
                                        OpenInferenceMimeTypeValues.JSON.value,
                                    )
                                    # Add gen_ai.output.messages
                                    if hasattr(event, "content") and event.content:
                                        try:
                                            if (
                                                hasattr(event.content, "parts")
                                                and event.content.parts
                                            ):
                                                for i, part in enumerate(event.content.parts):
                                                    if hasattr(part, "text") and part.text:
                                                        span.set_attribute(
                                                            f"gen_ai.output.messages.0.content.{i}.text",
                                                            part.text,
                                                        )
                                        except Exception:
                                            logger.exception(
                                                "Failed to get gen_ai.output.messages attribute."
                                            )
                                except Exception:
                                    logger.exception(
                                        f"Failed to get attribute: {SpanAttributes.OUTPUT_VALUE}."
                                    )
                            yield event
                        span.set_status(StatusCode.OK)
                    except Exception as e:
                        error_type = type(e).__name__
                        raise
                    finally:
                        # Record workflow duration metric
                        if workflow_duration_histogram:
                            duration = time.time() - start_time
                            metric_attributes = {"gen_ai.workflow.name": workflow_name}
                            if error_type:
                                metric_attributes["error.type"] = error_type
                            workflow_duration_histogram.record(duration, metric_attributes)

        return _AsyncGenerator(generator)


class _BaseAgentRunAsync(_WithTracer):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        meter: Optional[Meter] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(tracer, meter, *args, **kwargs)
        self._agent_duration_histogram = None
        if self._meter:
            self._agent_duration_histogram = self._meter.create_histogram(
                name="gen_ai.agent.duration",
                description="Duration of agent operations",
                unit="s",
            )

    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Event, None]],
        instance: BaseAgent,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        meter = self._meter
        agent_duration_histogram = self._agent_duration_histogram if self._meter else None
        agent_name = instance.name
        name = f"agent_run [{agent_name}]"
        attributes = dict(get_attributes_from_context())
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT.value
        attributes[SpanAttributes.AGENT_NAME] = agent_name

        # Add gen_ai span attributes
        attributes["gen_ai.operation.name"] = "agent_run"
        # Use agent name as agent ID if available
        if hasattr(instance, "id"):
            attributes["gen_ai.agent.id"] = str(instance.id)
        else:
            attributes["gen_ai.agent.id"] = agent_name

        class _AsyncGenerator(wrapt.ObjectProxy):  # type: ignore[misc]
            __wrapped__: AsyncGenerator[Event, None]

            async def __aiter__(self) -> Any:
                start_time = time.time()
                error_type = None
                with tracer.start_as_current_span(
                    name=name,
                    attributes=attributes,
                ) as span:
                    try:
                        async for event in self.__wrapped__:
                            if event.is_final_response():
                                try:
                                    output_value = event.model_dump_json(exclude_none=True)
                                    span.set_attribute(
                                        SpanAttributes.OUTPUT_VALUE,
                                        output_value,
                                    )
                                    span.set_attribute(
                                        SpanAttributes.OUTPUT_MIME_TYPE,
                                        OpenInferenceMimeTypeValues.JSON.value,
                                    )
                                    # Add gen_ai.output.messages
                                    if hasattr(event, "content") and event.content:
                                        try:
                                            if (
                                                hasattr(event.content, "parts")
                                                and event.content.parts
                                            ):
                                                for i, part in enumerate(event.content.parts):
                                                    if hasattr(part, "text") and part.text:
                                                        span.set_attribute(
                                                            f"gen_ai.output.messages.0.content.{i}.text",
                                                            part.text,
                                                        )
                                        except Exception:
                                            logger.exception(
                                                "Failed to get gen_ai.output.messages attribute."
                                            )
                                except Exception:
                                    logger.exception(
                                        f"Failed to get attribute: {SpanAttributes.OUTPUT_VALUE}."
                                    )
                            yield event
                        span.set_status(StatusCode.OK)
                    except Exception as e:
                        error_type = type(e).__name__
                        raise
                    finally:
                        # Record agent duration metric
                        if agent_duration_histogram:
                            duration = time.time() - start_time
                            metric_attributes = {"gen_ai.operation.name": "agent_run"}
                            if error_type:
                                metric_attributes["error.type"] = error_type
                            agent_duration_histogram.record(duration, metric_attributes)

        return _AsyncGenerator(generator)


class _TraceCallLlm(_WithTracer):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        meter: Meter,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(tracer, *args, **kwargs)
        self._meter = meter

        # Create histogram instruments for metrics
        self._time_to_first_token_histogram = self._meter.create_histogram(
            name="gen_ai.client.time_to_first_token",
            description="Time to first token for LLM operations",
            unit="s",
        )
        self._time_per_output_token_histogram = self._meter.create_histogram(
            name="gen_ai.client.time_per_output_token",
            description="Time per output token for LLM operations",
            unit="s",
        )
        self._time_between_token_histogram = self._meter.create_histogram(
            name="gen_ai.client.time_between_token",
            description="Time between consecutive tokens for LLM operations",
            unit="s",
        )
        self._operation_duration_histogram = self._meter.create_histogram(
            name="gen_ai.client.operation.duration",
            description="Duration of LLM operations",
            unit="s",
        )
        self._token_usage_histogram = self._meter.create_histogram(
            name="gen_ai.client.token.usage",
            description="Token usage for LLM operations",
            unit="{token}",
        )
        self._cached_tokens_histogram = self._meter.create_histogram(
            name="gen_ai.usage.prompt_tokens_details.cached_tokens",
            description="Number of cached tokens in prompt",
            unit="{token}",
        )
        self._operation_histogram = self._meter.create_histogram(
            name="gen_ai.client.operation",
            description="LLM operation type",
            unit="1",
        )

    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., T],
        _: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> T:
        # Record start time for operation duration
        start_time = time.time()

        ans = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return ans

        # Record end time and calculate operation duration
        end_time = time.time()
        operation_duration = end_time - start_time

        span = get_current_span()
        span.set_status(StatusCode.OK)  # Pre-emptively set status to OK
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        llm_request = next((arg for arg in arguments.values() if isinstance(arg, LlmRequest)), None)
        llm_response = next(
            (arg for arg in arguments.values() if isinstance(arg, LlmResponse)), None
        )

        # Prepare attributes for metrics
        metric_attributes = {}
        if llm_request and llm_request.model:
            metric_attributes["gen_ai.request.model"] = llm_request.model

        # Record operation duration metric
        self._operation_duration_histogram.record(operation_duration, metric_attributes)

        input_messages_index = 0
        if llm_request:
            span.set_attribute(
                SpanAttributes.LLM_PROVIDER,
                OpenInferenceLLMProviderValues.GOOGLE.value,
            )  # TODO: other providers may also be possible

            try:
                span.set_attribute(
                    SpanAttributes.INPUT_VALUE,
                    llm_request.model_dump_json(exclude_none=True, fallback=_default),
                )
                span.set_attribute(
                    SpanAttributes.INPUT_MIME_TYPE,
                    OpenInferenceMimeTypeValues.JSON.value,
                )
            except Exception:
                logger.exception(f"Failed to get attribute: {SpanAttributes.INPUT_VALUE}.")

            if llm_request.tools_dict:
                for i, tool in enumerate(llm_request.tools_dict.values()):
                    for k, v in _get_attributes_from_base_tool(
                        tool,
                        prefix=f"{SpanAttributes.LLM_TOOLS}.{i}.",
                    ):
                        span.set_attribute(k, v)

            if llm_request.model:
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, llm_request.model)

            if config := llm_request.config:
                for k, v in _get_attributes_from_generate_content_config(config):
                    span.set_attribute(k, v)

                if system_instruction := config.system_instruction:
                    span.set_attribute(
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_ROLE}",
                        "system",
                    )
                    if isinstance(system_instruction, str):
                        span.set_attribute(
                            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_CONTENT}",
                            system_instruction,
                        )
                    elif isinstance(system_instruction, types.Content):
                        if system_instruction.parts:
                            for k, v in _get_attributes_from_parts(
                                system_instruction.parts,
                                span_attribute=SpanAttributes.LLM_INPUT_MESSAGES,
                                message_index=input_messages_index,
                                text_only=True,
                            ):
                                span.set_attribute(k, v)
                    elif isinstance(system_instruction, list):
                        # TODO
                        pass
                    input_messages_index += 1

            if contents := llm_request.contents:
                for i, content in enumerate(contents, input_messages_index):
                    for k, v in _get_attributes_from_content(
                        content,
                        span_attribute=SpanAttributes.LLM_INPUT_MESSAGES,
                        message_index=i,
                    ):
                        span.set_attribute(k, v)
        if llm_response:
            for k, v in _get_attributes_from_llm_response(llm_response):
                span.set_attribute(k, v)

            # Record metrics from llm_response
            if llm_response.usage_metadata:
                usage = llm_response.usage_metadata

                # Record total token usage
                if usage.total_token_count:
                    self._token_usage_histogram.record(
                        usage.total_token_count, {**metric_attributes, "gen_ai.token.type": "total"}
                    )

                # Record prompt token usage
                if usage.prompt_token_count:
                    self._token_usage_histogram.record(
                        usage.prompt_token_count,
                        {**metric_attributes, "gen_ai.token.type": "input"},
                    )

                # Record completion token usage
                completion_tokens = 0
                if usage.candidates_token_count:
                    completion_tokens += usage.candidates_token_count
                if usage.thoughts_token_count:
                    completion_tokens += usage.thoughts_token_count
                if completion_tokens > 0:
                    self._token_usage_histogram.record(
                        completion_tokens, {**metric_attributes, "gen_ai.token.type": "output"}
                    )

                # Record cached tokens if available
                if usage.prompt_tokens_details:
                    cached_tokens = 0
                    for modality_token_count in usage.prompt_tokens_details:
                        # Check if there's a cached_token_count attribute
                        if (
                            hasattr(modality_token_count, "cached_token_count")
                            and modality_token_count.cached_token_count
                        ):
                            cached_tokens += modality_token_count.cached_token_count
                    if cached_tokens > 0:
                        self._cached_tokens_histogram.record(cached_tokens, metric_attributes)

                # Calculate time per output token
                if completion_tokens > 0 and operation_duration > 0:
                    time_per_token = operation_duration / completion_tokens
                    self._time_per_output_token_histogram.record(time_per_token, metric_attributes)

                    # NOTE: For non-streaming LLM calls, we cannot measure individual token timing.
                    # The following metrics (time_to_first_token and time_between_token) are
                    # approximations based on average time per token. In streaming scenarios,
                    # these would be tracked with actual per-token timing measurements.
                    # These approximations provide baseline metrics for non-streaming calls.
                    self._time_to_first_token_histogram.record(time_per_token, metric_attributes)
                    self._time_between_token_histogram.record(time_per_token, metric_attributes)

        # Record operation type (always 1 for chat completion)
        self._operation_histogram.record(1, {**metric_attributes, "gen_ai.operation.name": "chat"})

        return ans


class _TraceToolCall(_WithTracer):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        meter: Optional[Meter] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(tracer, meter, *args, **kwargs)
        self._tool_duration_histogram = None
        if self._meter:
            self._tool_duration_histogram = self._meter.create_histogram(
                name="gen_ai.tool.duration",
                description="Duration of tool operations",
                unit="s",
            )

    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., T],
        _: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> T:
        start_time = time.time()
        error_type = None
        tool_name = None

        try:
            ans = wrapped(*args, **kwargs)
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return ans
            span = get_current_span()
            span.set_status(StatusCode.OK)  # Pre-emptively set status to OK
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.TOOL.value,
            )

            # Add gen_ai span attributes
            span.set_attribute("gen_ai.operation.name", "tool_call")

            arguments = bind_args_kwargs(wrapped, *args, **kwargs)
            if base_tool := next(
                (arg for arg in arguments.values() if isinstance(arg, BaseTool)), None
            ):
                tool_name = base_tool.name
                span.set_attribute(SpanAttributes.TOOL_NAME, tool_name)
                span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, base_tool.description)

                # Add gen_ai.tool.name and gen_ai.tool.type
                span.set_attribute("gen_ai.tool.name", tool_name)
                span.set_attribute("gen_ai.tool.type", "function")

                if args_dict := next(
                    (arg for arg in arguments.values() if isinstance(arg, Mapping)), None
                ):
                    try:
                        tool_args_json = safe_json_dumps(args_dict)
                        span.set_attribute(
                            SpanAttributes.TOOL_PARAMETERS,
                            tool_args_json,
                        )
                        span.set_attribute(
                            SpanAttributes.INPUT_VALUE,
                            tool_args_json,
                        )
                        span.set_attribute(
                            SpanAttributes.INPUT_MIME_TYPE,
                            OpenInferenceMimeTypeValues.JSON.value,
                        )
                        # Add gen_ai.tool.call.arguments
                        span.set_attribute("gen_ai.tool.call.arguments", tool_args_json)
                    except Exception:
                        logger.exception(f"Failed to get attribute: {SpanAttributes.INPUT_VALUE}.")

            if event := next((arg for arg in arguments.values() if isinstance(arg, Event)), None):
                # Try to get tool call ID
                if hasattr(event, "content") and event.content:
                    try:
                        if hasattr(event.content, "parts") and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, "function_call") and part.function_call:
                                    if hasattr(part.function_call, "id") and part.function_call.id:
                                        span.set_attribute(
                                            "gen_ai.tool.call.id", part.function_call.id
                                        )
                                        break
                    except Exception:
                        logger.exception("Failed to get gen_ai.tool.call.id attribute.")

                if responses := event.get_function_responses():
                    try:
                        result_json = responses[0].model_dump_json(exclude_none=True)
                        span.set_attribute(
                            SpanAttributes.OUTPUT_VALUE,
                            result_json,
                        )
                        span.set_attribute(
                            SpanAttributes.OUTPUT_MIME_TYPE,
                            OpenInferenceMimeTypeValues.JSON.value,
                        )
                        # Add gen_ai.tool.call.result
                        span.set_attribute("gen_ai.tool.call.result", result_json)
                    except Exception:
                        logger.exception(f"Failed to get attribute in {wrapped.__name__}.")

            return ans
        except Exception as e:
            error_type = type(e).__name__
            raise
        finally:
            # Record tool duration metric
            if self._tool_duration_histogram and tool_name:
                duration = time.time() - start_time
                metric_attributes = {"gen_ai.tool.name": tool_name}
                if error_type:
                    metric_attributes["error.type"] = error_type
                self._tool_duration_histogram.record(duration, metric_attributes)


def stop_on_exception(
    wrapped: Callable[P, Iterator[tuple[str, AttributeValue]]],
) -> Callable[P, Iterator[tuple[str, AttributeValue]]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[tuple[str, AttributeValue]]:
        try:
            yield from wrapped(*args, **kwargs)
        except Exception:
            logger.exception(f"Failed to get attribute in {wrapped.__name__}.")

    return wrapper


@stop_on_exception
def _get_attributes_from_generate_content_config(
    obj: types.GenerateContentConfig,
) -> Iterator[tuple[str, AttributeValue]]:
    yield (
        SpanAttributes.LLM_INVOCATION_PARAMETERS,
        obj.model_dump_json(exclude_none=True, fallback=_default),
    )


@stop_on_exception
def _get_attributes_from_llm_response(
    obj: LlmResponse,
) -> Iterator[tuple[str, AttributeValue]]:
    yield SpanAttributes.OUTPUT_VALUE, obj.model_dump_json(exclude_none=True)
    yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
    if obj.usage_metadata:
        yield from _get_attributes_from_usage_metadata(obj.usage_metadata)
    if obj.content:
        yield from _get_attributes_from_content(
            obj.content, span_attribute=SpanAttributes.LLM_OUTPUT_MESSAGES, message_index=0
        )


@stop_on_exception
def _get_attributes_from_usage_metadata(
    obj: types.GenerateContentResponseUsageMetadata,
) -> Iterator[tuple[str, AttributeValue]]:
    if total := obj.total_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total
    if obj.prompt_tokens_details:
        prompt_details_audio = 0
        for modality_token_count in obj.prompt_tokens_details:
            if (
                modality_token_count.modality is types.MediaModality.AUDIO
                and modality_token_count.token_count
            ):
                prompt_details_audio += modality_token_count.token_count
        if prompt_details_audio:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
                prompt_details_audio,
            )
    if prompt := obj.prompt_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt
    if obj.candidates_tokens_details:
        completion_details_audio = 0
        for modality_token_count in obj.candidates_tokens_details:
            if (
                modality_token_count.modality is types.MediaModality.AUDIO
                and modality_token_count.token_count
            ):
                completion_details_audio += modality_token_count.token_count
        if completion_details_audio:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
                completion_details_audio,
            )
    completion = 0
    if candidates := obj.candidates_token_count:
        completion += candidates
    if thoughts := obj.thoughts_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, thoughts
        completion += thoughts
    if completion:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion


@stop_on_exception
def _get_attributes_from_content(
    obj: types.Content,
    /,
    *,
    span_attribute: str = SpanAttributes.LLM_INPUT_MESSAGES,
    message_index: int = 0,
) -> Iterator[tuple[str, AttributeValue]]:
    role = obj.role or "user"
    prefix = f"{span_attribute}.{message_index}."
    yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", role
    if parts := obj.parts:
        yield from _get_attributes_from_parts(
            parts, span_attribute=span_attribute, message_index=message_index
        )


@stop_on_exception
def _get_attributes_from_parts(
    obj: Iterable[types.Part],
    /,
    *,
    span_attribute: str = SpanAttributes.LLM_INPUT_MESSAGES,
    message_index: int = 0,
    text_only: bool = False,
) -> Iterator[tuple[str, AttributeValue]]:
    for i, part in enumerate(obj):
        if (text := part.text) is not None:
            prefix = f"{span_attribute}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{i}."
            yield from _get_attributes_from_text_part(
                text,
                prefix=prefix,
            )
        elif text_only:
            continue
        elif (function_call := part.function_call) is not None:
            prefix = f"{span_attribute}.{message_index}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{i}."
            yield from _get_attributes_from_function_call(
                function_call,
                prefix=prefix,
            )
        elif (function_response := part.function_response) is not None:
            prefix = f"{span_attribute}.{message_index}."
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "tool"
            if function_response.name:
                yield f"{prefix}{MessageAttributes.MESSAGE_NAME}", function_response.name
            if function_response.response:
                yield (
                    f"{prefix}{MessageAttributes.MESSAGE_CONTENT}",
                    safe_json_dumps(function_response.response),
                )
            message_index += 1


@stop_on_exception
def _get_attributes_from_text_part(
    obj: str,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", obj
    yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"


@stop_on_exception
def _get_attributes_from_function_call(
    obj: types.FunctionCall,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    if id_ := obj.id:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id_
    if name := obj.name:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", name
    if function_arguments := obj.args:
        yield (
            f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
            safe_json_dumps(function_arguments),
        )


@stop_on_exception
def _get_attributes_from_function_response(
    obj: types.FunctionResponse,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    if id_ := obj.id:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id_
    if name := obj.name:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", name
    if response := obj.response:
        yield (
            f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
            safe_json_dumps(response),
        )


@stop_on_exception
def _get_attributes_from_base_tool(
    obj: BaseTool,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    if declaration := obj._get_declaration():
        tool_json_schema = declaration.model_dump_json(exclude_none=True)
    else:
        tool_json_schema = json.dumps({"name": obj.name, "description": obj.description})
    yield f"{prefix}{ToolAttributes.TOOL_JSON_SCHEMA}", tool_json_schema


def bind_args_kwargs(func: Any, *args: Any, **kwargs: Any) -> OrderedDict[str, Any]:
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _default(obj: Any) -> Any:
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    if inspect.isclass(obj) and issubclass(obj, BaseModel):
        return obj.model_json_schema()
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode()
    return str(obj)
