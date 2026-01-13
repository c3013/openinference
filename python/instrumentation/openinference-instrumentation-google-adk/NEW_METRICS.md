# New Gen AI Metrics in Google ADK Instrumentation

This document describes the gen_ai metrics added to the Google ADK instrumentation plugin. Token usage metrics are reported as OpenTelemetry histogram metrics, while timing metrics are captured as span attributes for trace correlation.

## Overview

The following metrics are tracked during LLM interactions:

### Histogram Metrics (LLM Spans Only)

These metrics are recorded as OpenTelemetry histograms when LLM calls are made:

#### 1. gen_ai.client.token.usage

**Type:** Histogram (Integer)  
**Description:** Total number of tokens used in the operation (prompt + completion).  
**Use Case:** Monitor token consumption for cost tracking and quota management.  
**Attributes:** `gen_ai.client.operation` (set to "chat")
**Span Type:** LLM

#### 2. gen_ai.usage.prompt_tokens_details.cached_tokens

**Type:** Histogram (Integer)  
**Description:** Number of prompt tokens that were cached and reused from previous requests.  
**Use Case:** Track cache hit rate and cost savings from prompt caching.  
**Attributes:** `gen_ai.client.operation` (set to "chat")
**Span Type:** LLM

### Span Attributes (Timing Information)

These metrics are captured as span attributes for trace correlation but not reported as separate histogram metrics:

#### 3. gen_ai.client.time_to_first_token

**Type:** Float (milliseconds)  
**Description:** Time elapsed from the start of the request to receiving the first token in the response.  
**Use Case:** Measure initial latency and response time. Useful for understanding user-perceived latency.
**Span Type:** Chain (invocation level)

#### 4. gen_ai.client.time_per_output_token

**Type:** Float (milliseconds)  
**Description:** Average time taken per output token, calculated as total generation time divided by token count.  
**Use Case:** Measure generation throughput. Helps understand overall generation speed.
**Span Type:** Chain (invocation level)

#### 5. gen_ai.client.time_between_token

**Type:** Float (milliseconds)  
**Description:** Time elapsed between consecutive tokens during streaming.  
**Use Case:** Monitor token generation consistency. Can help identify stuttering or delays in streaming.
**Span Type:** Chain (invocation level)

#### 6. gen_ai.client.operation.duration

**Type:** Float (milliseconds)  
**Description:** Total duration of the operation from start to finish.  
**Use Case:** Track end-to-end operation latency for performance monitoring and optimization.
**Span Type:** Chain (invocation level)

#### 7. gen_ai.client.operation

**Type:** String attribute  
**Description:** The type of operation being performed. Currently set to "chat" for LLM interactions.  
**Use Case:** Used as a dimension/label for filtering and grouping histogram metrics.
**Span Type:** LLM

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

Histogram metrics are automatically collected when using the Google ADK instrumentor with a meter provider:

```python
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry import metrics

# Set up meter provider (use your observability backend's meter provider)
meter_provider = metrics.get_meter_provider()

# Instrument with meter provider to enable histogram metrics
GoogleADKInstrumentor().instrument(meter_provider=meter_provider)

# Your Google ADK code here
# Token usage metrics will be automatically recorded as histograms for LLM calls
# Timing metrics will be available as span attributes for trace correlation
```

## Metrics and Span Types

### LLM Span Metrics (Histograms)

These histograms are recorded only when actual LLM calls are made:

- **gen_ai.client.token.usage**: Total token count
- **gen_ai.usage.prompt_tokens_details.cached_tokens**: Cached token count

Both include the `gen_ai.client.operation: "chat"` attribute.

### Chain Span Attributes (Timing)

These are captured as span attributes at the invocation/chain level for trace correlation:

- **gen_ai.client.time_to_first_token**: Latency to first token (ms)
- **gen_ai.client.time_per_output_token**: Average time per token (ms)
- **gen_ai.client.time_between_token**: Inter-token timing (ms)
- **gen_ai.client.operation.duration**: Total operation duration (ms)

## Example

After instrumentation, metrics are recorded as follows:

**LLM Span** (histogram metrics):
```python
# Recorded as histograms during LLM calls:
# - gen_ai.client.token.usage: 350 tokens (with operation="chat")
# - gen_ai.usage.prompt_tokens_details.cached_tokens: 150 tokens (with operation="chat")
```

**Chain Span** (span attributes only):
```python
# Available as span attributes for trace correlation:
# - gen_ai.client.time_to_first_token: 234.5 ms
# - gen_ai.client.time_per_output_token: 12.3 ms
# - gen_ai.client.time_between_token: 15.7 ms
# - gen_ai.client.operation.duration: 1523.8 ms
```

The histogram metrics can be:
- Exported to observability platforms (Prometheus, Datadog, etc.)
- Analyzed with percentile aggregation (P50, P95, P99)
- Used for alerting and performance monitoring

The span attributes can be:
- Correlated with trace data
- Used for detailed performance analysis
- Filtered and grouped in trace visualization tools

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
