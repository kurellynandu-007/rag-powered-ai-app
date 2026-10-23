"""Send a minimal request to verify local GroqCloud configuration."""

import sys

from groq import APIError

from config import (
    GroqConfigurationError,
    create_groq_client,
    get_groq_model,
)


def main() -> int:
    try:
        model = get_groq_model()
        client = create_groq_client()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=10,
        )
    except GroqConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    except APIError as error:
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            print(
                "Groq request failed. Check your network and GroqCloud "
                "configuration; no credential details were displayed.",
                file=sys.stderr,
            )
        else:
            print(
                f"Groq request failed with HTTP status {status_code}. "
                "Check the selected model and local credentials; "
                "no credential details were displayed.",
                file=sys.stderr,
            )
        return 1

    print("Groq test request succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
