"""HuggingFace Inference API explanation module.

Calls Mistral-7B-Instruct via the HF Inference API to produce 2-sentence
DevOps failure explanations, with fallback templates for API unavailability.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
REQUEST_TIMEOUT = 10

_explanation_cache: dict[str, str] = {}

FALLBACK_TEMPLATES = {
    "build_failure": (
        "The pipeline failed during the build stage due to a dependency resolution or compilation "
        "error — a missing package, version conflict, or syntax issue prevented the build from completing. "
        "Review the dependency manifests (package.json, requirements.txt, pom.xml) and ensure all "
        "declared versions are available and compatible."
    ),
    "test_failure": (
        "The test suite reported one or more failures, indicating a regression in application logic, "
        "an environment mismatch, or insufficient test coverage that fell below the required threshold. "
        "Inspect the failing test assertions and confirm that the application behaves as expected "
        "in the CI environment."
    ),
    "timeout": (
        "The pipeline job was terminated because it exceeded the configured execution time limit, "
        "likely caused by a hung process, slow external dependency, or an infinite loop in a build step. "
        "Investigate the last logged step before the timeout and consider adding explicit timeouts "
        "to long-running sub-tasks."
    ),
    "config_error": (
        "The pipeline failed at the configuration stage because a required environment variable, "
        "secret, or configuration file was missing or malformed, preventing the application from "
        "initialising. Verify that all required secrets are injected into the CI environment and "
        "that configuration files are syntactically valid."
    ),
    "infra_failure": (
        "The deployment failed due to an infrastructure-level issue — such as an OOM kill, "
        "image pull error, insufficient cluster resources, or a storage mount failure — that "
        "prevented the workload from starting. Check the Kubernetes events and node conditions, "
        "then review resource requests/limits and image registry access."
    ),
}

VALID_LABELS = set(FALLBACK_TEMPLATES.keys())

PROMPT_TEMPLATE = (
    "[INST] You are a DevOps expert. Analyze this CI/CD pipeline log and explain in exactly "
    "2 sentences why it failed. Predicted category: {label}.\n\n"
    "Log:\n{log}\n\nExplanation: [/INST]"
)


def _call_hf_api(prompt: str, hf_token: str) -> str:
    """Call the HuggingFace Inference API and return the generated text.

    Args:
        prompt: Fully formatted prompt string.
        hf_token: Bearer token for authentication.

    Returns:
        Generated explanation string from the model.

    Raises:
        requests.RequestException: On network or HTTP error.
        ValueError: If the API response cannot be parsed.
    """
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.3,
            "return_full_text": False,
            "stop": ["\n\n", "[INST]"],
        },
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    raise ValueError(f"Unexpected API response structure: {data}")


def explain_failure(log_text: str, predicted_label: str) -> str:
    """Generate a 2-sentence explanation of a CI/CD failure.

    Uses the HuggingFace Inference API with Mistral-7B-Instruct. Falls back to
    a static template per label if the API is unavailable or times out.
    Results are cached in-memory for the duration of the process.

    Args:
        log_text: Raw multi-line CI/CD log snippet.
        predicted_label: Predicted failure category (one of the 5 known labels).

    Returns:
        A 2-sentence explanation string.
    """
    if predicted_label not in VALID_LABELS:
        predicted_label = "infra_failure"

    cache_key = f"{predicted_label}::{log_text[:200]}"
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: HF_TOKEN not set — using fallback explanation."
        )
        explanation = FALLBACK_TEMPLATES[predicted_label]
        _explanation_cache[cache_key] = explanation
        return explanation

    truncated_log = log_text[:1200] if len(log_text) > 1200 else log_text
    prompt = PROMPT_TEMPLATE.format(label=predicted_label, log=truncated_log)

    try:
        explanation = _call_hf_api(prompt, hf_token)
        if len(explanation) < 20:
            raise ValueError("Response too short, falling back.")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] HF API explanation received ({len(explanation)} chars).")
    except requests.exceptions.Timeout:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: HF API timed out after {REQUEST_TIMEOUT}s — using fallback."
        )
        explanation = FALLBACK_TEMPLATES[predicted_label]
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response else "?"
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: HF API HTTP {status} — using fallback."
        )
        explanation = FALLBACK_TEMPLATES[predicted_label]
    except Exception as exc:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: HF API error ({exc}) — using fallback."
        )
        explanation = FALLBACK_TEMPLATES[predicted_label]

    _explanation_cache[cache_key] = explanation
    return explanation


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    sample_log = (
        "npm ERR! code ENOENT\n"
        "npm ERR! syscall open\n"
        "npm ERR! path /home/runner/work/myapp/package.json\n"
        "npm ERR! enoent ENOENT: no such file or directory\n"
        "Build failed with exit code 1"
    )
    result = explain_failure(sample_log, "build_failure")
    print(f"\nExplanation:\n{result}")
