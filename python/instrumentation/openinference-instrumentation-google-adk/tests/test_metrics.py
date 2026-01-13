# ruff: noqa: E501
from secrets import token_hex
from typing import Any

import pytest
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
    decode_compressed_response=True,
)
async def test_google_adk_metrics(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    """Test that metrics are recorded for LLM operations."""

    def get_weather(city: str) -> dict[str, str]:
        """Retrieves the current weather report for a specified city.

        Args:
            city (str): The name of the city for which to retrieve the weather report.

        Returns:
            dict: status and result or error msg.
        """
        return {
            "status": "success",
            "report": (
                f"The weather in {city} is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }

    agent_name = f"_{token_hex(4)}"
    agent = Agent(
        name=agent_name,
        model="gemini-2.0-flash",
        description="Agent to answer questions using tools.",
        instruction="You must use the available tools to find an answer.",
        tools=[get_weather],
    )

    app_name = token_hex(4)
    user_id = token_hex(4)
    session_id = token_hex(4)
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="What is the weather in New York?")]
        ),
    ):
        ...

    # Check that metrics were recorded
    metrics_data = in_memory_metric_reader.get_metrics_data()
    assert metrics_data is not None

    # Get all resource metrics
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    # Collect all metric names
    metric_names = set()
    for rm in resource_metrics:
        for scope_metric in rm.scope_metrics:
            for metric in scope_metric.metrics:
                metric_names.add(metric.name)

    # Verify expected metrics are present
    expected_metrics = {
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation",
    }

    # Check that at least some of the expected metrics are present
    found_metrics = expected_metrics & metric_names
    assert len(found_metrics) > 0, (
        f"Expected at least some metrics from {expected_metrics}, but found {metric_names}"
    )

    # Verify that operation.duration metric has data points
    for rm in resource_metrics:
        for scope_metric in rm.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    # Check that histogram has data points
                    assert hasattr(metric.data, "data_points")
                    assert len(metric.data.data_points) > 0
                    # Verify the data point has a value
                    for data_point in metric.data.data_points:
                        # For histogram data points, check count
                        if hasattr(data_point, "count"):
                            assert data_point.count > 0
