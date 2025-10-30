"""
Simple script to verify Supabase connectivity and basic queries.

Usage:
    python scripts/test_supabase_connection.py

Requirements:
    - .env with SUPABASE_URL and SUPABASE_SERVICE_KEY in project root
    - pip install supabase python-dotenv
"""

import os
import sys
import time
import socket
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from supabase import create_client, Client


def get_env_variable(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def normalize_supabase_url(raw_url: Optional[str], project_ref: Optional[str]) -> Optional[str]:
    """Return a valid Supabase URL or None.
    Accepts either a full URL or builds one from project ref.
    """
    if not raw_url and project_ref:
        raw_url = f"https://{project_ref}.supabase.co"

    if not raw_url:
        return None

    # Prepend https:// if scheme missing
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        raw_url = "https://" + raw_url

    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return raw_url


def connect_supabase() -> Client:
    load_dotenv()

    raw_url = get_env_variable("SUPABASE_URL")
    project_ref = get_env_variable("SUPABASE_PROJECT_REF")
    supabase_url = normalize_supabase_url(raw_url, project_ref)
    supabase_key = get_env_variable("SUPABASE_SERVICE_KEY")

    if not supabase_url:
        print("✗ Invalid or missing Supabase URL. Set SUPABASE_URL (e.g., https://xxxxxxxx.supabase.co)\n   or SUPABASE_PROJECT_REF (e.g., xxxxxxxx) in your .env")
        sys.exit(1)

    if not supabase_key:
        print("✗ Missing SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)

    try:
        client: Client = create_client(supabase_url, supabase_key)
        return client
    except Exception as exc:
        print(f"✗ Failed to create Supabase client: {exc}")
        sys.exit(1)


def main() -> None:
    print("Checking Supabase connectivity...")
    client = connect_supabase()
    print("✓ Client created")

    # Basic read test from reference table expected by schema
    max_retries = int(os.getenv("SUPABASE_RETRY_COUNT", "5"))
    backoff_sec = float(os.getenv("SUPABASE_RETRY_BACKOFF", "1.5"))
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.table("resource_types").select("id,name,unit,critical_threshold").limit(5).execute()
            rows = response.data or []
            print(f"✓ Query ok: resource_types (showing {len(rows)} rows)")
            for row in rows:
                print(f"  - {row['id']}: {row['name']} ({row['unit']}), threshold={row['critical_threshold']}")
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            # Transient DNS/network issues often raise socket.gaierror or similar
            print(f"Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                sleep_for = backoff_sec ** attempt
                print(f"… retrying in {sleep_for:.1f}s")
                time.sleep(sleep_for)

    if last_exc is not None:
        # Helpful diagnostics
        url = normalize_supabase_url(get_env_variable("SUPABASE_URL"), get_env_variable("SUPABASE_PROJECT_REF")) or ""
        host = urlparse(url).netloc
        print("\nDiagnostics:")
        print(f"  Host: {host or 'unknown'}")
        try:
            if host:
                resolved = socket.gethostbyname(host)
                print(f"  DNS resolution: {resolved}")
        except Exception as dns_exc:
            print(f"  DNS resolution error: {dns_exc}")
        print("  Tips: Check internet/VPN, DNS, firewall/proxy. Ensure NO_PROXY includes 'supabase.co' if using a proxy.")
        print("  You can tweak retries via SUPABASE_RETRY_COUNT and SUPABASE_RETRY_BACKOFF.")
        print("✗ Query failed on resource_types after retries.")
        sys.exit(1)

    # Lightweight write-then-delete sanity check on events table
    # Skips if env var is set to avoid writes in restricted environments
    if os.getenv("SKIP_WRITE_TEST", "false").lower() != "true":
        try:
            dummy_event = {
                "event_type": "healthcheck",
                "event_name": "supabase-connection-test",
                "affected_region": None,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "severity": "low",
                "impact_description": "Connectivity check"
            }
            insert_resp = client.table("events").insert(dummy_event).execute()
            inserted = (insert_resp.data or [])[0]
            print(f"✓ Insert ok: events id={inserted['id']}")

            # Clean up
            client.table("events").delete().eq("id", inserted["id"]).execute()
            print("✓ Cleanup ok: test event deleted")
        except Exception as exc:
            print(f"! Write test skipped/failed: {exc}")

    print("✓ Supabase connectivity looks good")


if __name__ == "__main__":
    main()


