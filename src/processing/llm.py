import time


def call_llm_with_retry(model, prompt, system_instruction, config):
    """Call LLM with retry logic."""
    for attempt in range(config["max_retries"]):
        try:
            result = model.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                config=config,
            )
            return result
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ["401", "403", "authentication", "invalid_api_key"]):
                return f"ERROR_FATAL: {e}"
            if attempt == config["max_retries"] - 1:
                return f"ERROR_RETRY: {e}"
            wait = 2 ** attempt + 1
            if any(kw in err for kw in ["429", "rate_limit", "resource_exhausted"]):
                wait += 10
            time.sleep(wait)
    return "ERROR_RETRY"
