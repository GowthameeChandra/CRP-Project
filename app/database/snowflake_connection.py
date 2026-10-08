# Personal test script - connects to Snowflake and reads SF_TEST_TABLE.
# Standalone usage: python app/database/snowflake_connection.py
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from snowflake.snowpark import Session

from snowflake_creds import (
    SF_ACCOUNT,
    SF_USER,
    SF_WAREHOUSE,
    SF_DATABASE,
    SF_SCHEMA,
    SF_ROLE,
    SF_TEST_TABLE,
    SF_PRIVATE_KEY_PATH,
    SF_PRIVATE_KEY_PASSPHRASE_PATH,
)
from snowflake_decrypt import extract_key_bytes, get_private_key


def main() -> None:
    connection_params = {
        "account": SF_ACCOUNT,
        "user": SF_USER,
        "private_key": extract_key_bytes(
            get_private_key(SF_PRIVATE_KEY_PATH, SF_PRIVATE_KEY_PASSPHRASE_PATH)
        ),
        "warehouse": SF_WAREHOUSE,
        "database": SF_DATABASE,
        "schema": SF_SCHEMA,
        "role": SF_ROLE,
    }

    print("Connecting to Snowflake...")
    session = Session.builder.configs(connection_params).create()
    print("Connected.")

    try:
        table = session.table(SF_TEST_TABLE)
        total_records = table.count()
        print(f"Total records in {SF_TEST_TABLE}: {total_records}")

        print("Showing first 10 rows:")
        table.limit(10).show()
    finally:
        session.close()
        print("Session closed.")


if __name__ == "__main__":
    main()
