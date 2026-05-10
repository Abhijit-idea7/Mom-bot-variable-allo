"""
accounts.py
-----------
Account registry for multi-account stocksdeveloper trading.

Add a new broker account by editing accounts.json — no Python changes needed:

  {
    "default": {
      "display_name": "Primary Account (Zerodha)",
      "api_key_env":  "STOCKSDEVELOPER_API_KEY",
      "account_id_env": "STOCKSDEVELOPER_ACCOUNT"
    },
    "abhi2": {
      "display_name": "Second Zerodha Account",
      "api_key_env":  "STOCKSDEVELOPER_API_KEY_ABHI2",
      "account_id_env": "STOCKSDEVELOPER_ACCOUNT_ABHI2"
    }
  }

Then add the corresponding secrets to GitHub Actions:
  STOCKSDEVELOPER_API_KEY_ABHI2  — second account API key
  STOCKSDEVELOPER_ACCOUNT_ABHI2  — second account ID (e.g. "AbhiZerodha2")

Each account's positions and trades are tracked in separate CSV files:
  account="default" → positions_nifty200.csv          (backward compat)
  account="abhi2"   → positions_nifty200_abhi2.csv
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from config import STOCKSDEVELOPER_ACCOUNT, STOCKSDEVELOPER_API_KEY

logger = logging.getLogger(__name__)

_ACCOUNTS_FILE = Path(__file__).parent / "accounts.json"


@dataclass(frozen=True)
class AccountConfig:
    name:         str  # key in accounts.json (e.g. "default", "abhi2")
    display_name: str  # human-readable label
    api_key:      str  # resolved from env at startup
    account_id:   str  # resolved from env at startup


def _load_accounts() -> dict:
    if not _ACCOUNTS_FILE.exists():
        return {}
    with _ACCOUNTS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_account(name: str = "default", dry_run: bool = False) -> AccountConfig:
    """Return an AccountConfig for the given account name.

    Reads API key and account ID from the environment variables named in
    accounts.json.  Falls back to STOCKSDEVELOPER_API_KEY /
    STOCKSDEVELOPER_ACCOUNT from config.py for the "default" account when
    accounts.json does not exist (backward compatibility).

    In dry-run mode, a missing API key is replaced with a placeholder so
    the workflow can preview trades without secrets configured.
    """
    accounts = _load_accounts()
    key = (name or "default").strip().lower()

    if not accounts:
        # No accounts.json — use config.py legacy values
        api_key = STOCKSDEVELOPER_API_KEY
        if not api_key:
            if dry_run:
                api_key = "dry-run-key"
                logger.warning("  STOCKSDEVELOPER_API_KEY not set — using placeholder (dry-run)")
            else:
                raise EnvironmentError(
                    "STOCKSDEVELOPER_API_KEY is not set. "
                    "Add it to your .env file or GitHub Actions secrets."
                )
        return AccountConfig(
            name         = "default",
            display_name = "Primary Account",
            api_key      = api_key,
            account_id   = STOCKSDEVELOPER_ACCOUNT or "default",
        )

    if key not in accounts:
        raise ValueError(
            f"Unknown account '{name}'. "
            f"Valid choices: {list(accounts.keys())} — see accounts.json"
        )

    entry       = accounts[key]
    api_key_var = entry["api_key_env"]
    account_var = entry["account_id_env"]

    api_key    = os.getenv(api_key_var)
    account_id = os.getenv(account_var)

    if not api_key:
        if dry_run:
            api_key = "dry-run-key"
            logger.warning(f"  {api_key_var} not set — using placeholder (dry-run mode)")
        else:
            raise EnvironmentError(
                f"Environment variable '{api_key_var}' is not set. "
                f"Add it to .env or GitHub Actions secrets for account '{key}'."
            )

    if not account_id:
        raise EnvironmentError(
            f"Environment variable '{account_var}' is not set. "
            f"Add it to .env or GitHub Actions secrets for account '{key}'."
        )

    return AccountConfig(
        name         = key,
        display_name = entry.get("display_name", key),
        api_key      = api_key,
        account_id   = account_id,
    )


def list_accounts() -> list[str]:
    """Return all account names defined in accounts.json, or ['default'] if absent."""
    accounts = _load_accounts()
    return list(accounts.keys()) if accounts else ["default"]
