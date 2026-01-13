# New Gen AI Metrics in Google ADK Instrumentation

This document describes the 5 new gen_ai metrics added to the Google ADK instrumentation plugin.

## Overview

The following metrics are now automatically tracked and added as span attributes during LLM interactions:

### 1. gen_ai.client.time_to_first_token

**Type:** Float (milliseconds)  
**Description:** Time elapsed from the start of the request to receiving the first token in the response.  
**Use Case:** Measure initial latency and response time. Useful for understanding user-perceived latency.

### 2. gen_ai.client.time_per_output_token

**Type:** Float (milliseconds)  
**Description:** Average time taken per output token, calculated as total generation time divided by token count.  
**Use Case:** Measure generation throughput. Helps understand overall generation speed.

### 3. gen_ai.client.time_between_token

**Type:** Float (milliseconds)  
**Description:** Time elapsed between consecutive tokens during streaming.  
**Use Case:** Monitor token generation consistency. Can help identify stuttering or delays in streaming.

### 4. gen_ai.client.operation

**Type:** String  
**Description:** The type of operation being performed. Currently set to "chat" for LLM interactions.  
**Use Case:** Categorize different types of LLM operations for filtering and analysis.

### 5. gen_ai.usage.prompt_tokens_details.cached_tokens

**Type:** Integer  
**Description:** Number of prompt tokens that were cached and reused from previous requests.  
**Use Case:** Track cache hit rate and cost savings from prompt caching.

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

These metrics are automatically collected when using the Google ADK instrumentor. No additional configuration is required:

```python
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

GoogleADKInstrumentor().instrument()

# Your Google ADK code here
# The metrics will be automatically added to spans
```

## Metric Format

All metrics follow the OpenTelemetry semantic conventions for Gen AI observability and are added as span attributes that can be:

- Exported to observability platforms
- Analyzed with histogram aggregation
- Used for performance monitoring and alerting
- Correlated with other trace data

## Example Span Attributes

After instrumentation, a typical LLM span will include:

```json
{
  "gen_ai.client.time_to_first_token": 234.5,
  "gen_ai.client.time_per_output_token": 12.3,
  "gen_ai.client.time_between_token": 15.7,
  "gen_ai.client.operation": "chat",
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
