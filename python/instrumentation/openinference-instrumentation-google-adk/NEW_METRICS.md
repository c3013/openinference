# New Gen AI Metrics in Google ADK Instrumentation

This document describes the 7 gen_ai metrics added to the Google ADK instrumentation plugin. These metrics are reported as OpenTelemetry histogram metrics in addition to being set as span attributes.

## Overview

The following metrics are now automatically tracked and reported as OpenTelemetry histograms during LLM interactions:

### 1. gen_ai.client.time_to_first_token

**Type:** Histogram (Float, milliseconds)  
**Description:** Time elapsed from the start of the request to receiving the first token in the response.  
**Use Case:** Measure initial latency and response time. Useful for understanding user-perceived latency.  
**Attributes:** `gen_ai.client.operation` (set to "chat")

### 2. gen_ai.client.time_per_output_token

**Type:** Histogram (Float, milliseconds)  
**Description:** Average time taken per output token, calculated as total generation time divided by token count.  
**Use Case:** Measure generation throughput. Helps understand overall generation speed.  
**Attributes:** `gen_ai.client.operation` (set to "chat")

### 3. gen_ai.client.time_between_token

**Type:** Histogram (Float, milliseconds)  
**Description:** Time elapsed between consecutive tokens during streaming.  
**Use Case:** Monitor token generation consistency. Can help identify stuttering or delays in streaming.  
**Attributes:** `gen_ai.client.operation` (set to "chat")

### 4. gen_ai.client.operation

**Type:** String attribute (not a histogram)  
**Description:** The type of operation being performed. Currently set to "chat" for LLM interactions.  
**Use Case:** Used as a dimension/label for filtering and grouping other histogram metrics.

### 5. gen_ai.client.operation.duration

**Type:** Histogram (Float, milliseconds)  
**Description:** Total duration of the operation from start to finish.  
**Use Case:** Track end-to-end operation latency for performance monitoring and optimization.  
**Attributes:** `gen_ai.client.operation` (set to "chat")

### 6. gen_ai.client.token.usage

**Type:** Histogram (Integer)  
**Description:** Total number of tokens used in the operation (prompt + completion).  
**Use Case:** Monitor token consumption for cost tracking and quota management.  
**Attributes:** `gen_ai.client.operation` (set to "chat")

### 7. gen_ai.usage.prompt_tokens_details.cached_tokens

**Type:** Histogram (Integer)  
**Description:** Number of prompt tokens that were cached and reused from previous requests.  
**Use Case:** Track cache hit rate and cost savings from prompt caching.  
**Attributes:** `gen_ai.client.operation` (set to "chat")

## Implementation Details

### Timing Metrics

The timing metrics are captured during streaming responses:

```python
# Simplified implementation
start_time = time.time()
first_token_time = None
last_token_time = None

async for event in stream:
    current_time = time.time()
    
    if first_token_time is None and event.content:
        first_token_time = current_time
        time_to_first_token = (first_token_time - start_time) * 1000
        span.set_attribute("gen_ai.client.time_to_first_token", time_to_first_token)
    
    if last_token_time is not None:
        time_between_tokens = (current_time - last_token_time) * 1000
        span.set_attribute("gen_ai.client.time_between_token", time_between_tokens)
    
    last_token_time = current_time
```

### Cached Tokens

Cached tokens are extracted from the usage metadata in two ways:

1. From `cached_input_token_count` at the top level
2. From `cached_token_count` within each modality token count

```python
if hasattr(obj, 'cached_input_token_count') and obj.cached_input_token_count:
    yield ("gen_ai.usage.prompt_tokens_details.cached_tokens", obj.cached_input_token_count)

# Or from modality token counts
for modality_token_count in obj.prompt_tokens_details:
    if hasattr(modality_token_count, 'cached_token_count'):
        cached_tokens += modality_token_count.cached_token_count
```

## Usage

These metrics are automatically collected when using the Google ADK instrumentor:

```python
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry import metrics

# Set up meter provider (use your observability backend's meter provider)
meter_provider = metrics.get_meter_provider()

# Instrument with meter provider to enable metrics
GoogleADKInstrumentor().instrument(meter_provider=meter_provider)

# Your Google ADK code here
# The metrics will be automatically recorded as histograms
```

## Metric Format

All numeric metrics are recorded as OpenTelemetry histograms with the following characteristics:

- **Histogram Type**: Values are recorded with histogram instruments for aggregation
- **Attributes**: Each histogram includes `gen_ai.client.operation` attribute for filtering
- **Units**: Timing metrics are in milliseconds (ms), token metrics are in token counts
- **Span Attributes**: Values are also set as span attributes for correlation with traces

## Example Usage

After instrumentation, metrics are automatically recorded during LLM operations:

```python
# Metrics are recorded as histograms:
# - gen_ai.client.time_to_first_token: 234.5 ms (with operation="chat")
# - gen_ai.client.time_per_output_token: 12.3 ms (with operation="chat")
# - gen_ai.client.time_between_token: 15.7 ms (with operation="chat")
# - gen_ai.client.operation.duration: 1523.8 ms (with operation="chat")
# - gen_ai.client.token.usage: 350 tokens (with operation="chat")
# - gen_ai.usage.prompt_tokens_details.cached_tokens: 150 tokens (with operation="chat")
```

These histograms can be:
- Exported to observability platforms (Prometheus, Datadog, etc.)
- Analyzed with percentile aggregation (P50, P95, P99)
- Used for alerting and performance monitoring
- Correlated with trace data via span attributes

## Example Span Attributes

After instrumentation, a typical LLM span will include:

```json
{
  "gen_ai.client.time_to_first_token": 234.5,
  "gen_ai.client.time_per_output_token": 12.3,
  "gen_ai.client.time_between_token": 15.7,
  "gen_ai.client.operation": "chat",
  "gen_ai.client.operation.duration": 1523.8,
  "gen_ai.client.token.usage": 350,
  "gen_ai.usage.prompt_tokens_details.cached_tokens": 150,
  // ... other existing attributes
}
```

## Testing

Tests have been added in `tests/test_new_metrics.py` to verify:
- Attribute names are correctly defined
- Cached tokens extraction works for different scenarios
- Operation attribute is set correctly
- All timing metrics follow the expected format
