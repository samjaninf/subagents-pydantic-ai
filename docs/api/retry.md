# Retry API

Auto-retry for transient model and gateway failures. Each retry resumes from the
accumulated message history rather than starting over. See
[Retries](../advanced/retries.md).

## RetryConfig

::: subagents_pydantic_ai.RetryConfig
    options:
      show_root_heading: true
      show_source: true

## run_with_retry

::: subagents_pydantic_ai.run_with_retry
    options:
      show_root_heading: true
      show_source: true

## is_transient_error

::: subagents_pydantic_ai.is_transient_error
    options:
      show_root_heading: true
      show_source: true

## compute_backoff_delay

::: subagents_pydantic_ai.compute_backoff_delay
    options:
      show_root_heading: true
      show_source: true
