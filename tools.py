import logging
import os
import re
import struct
import sys

from pydantic import Field
import pyodbc
from contextlib import closing
from typing import Any

from azure.identity import DefaultAzureCredential, AzureCliCredential


# # Load environment variables from .env file (for local development)
# load_dotenv()

# # Get connection string from environment
# connection_string = os.environ.get("AZURE_SQL_CONNECTIONSTRING")
# print(f"Connected using: {connection_string}")
# if not connection_string:
#     raise ValueError("AZURE_SQL_CONNECTIONSTRING environment variable is required")


logger = logging.getLogger("tools")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default=%s", name, raw, default)
        return default


MAX_QUERY_RESULT_ROWS = _env_int("MAX_QUERY_RESULT_ROWS", 100)
MAX_QUERY_RESULT_CHARS = _env_int("MAX_QUERY_RESULT_CHARS", 12000)
MAX_SQL_CELL_CHARS = _env_int("MAX_SQL_CELL_CHARS", 1200)


def _get_token_struct(token: str) -> bytes:
    """Convert access token to the format required by pyodbc for SQL Server."""
    token_bytes = token.encode("utf-16-le")
    return struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)


def _truncate_text(value: str, max_chars: int, label: str) -> str:
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}\n...[TRUNCATED {label}: omitted {omitted} chars]"


def _sanitize_cell_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, MAX_SQL_CELL_CHARS, "SQL CELL")
    return value


class SqlDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._credential = None
        self._last_query_truncated = False
        # self._init_credential()
        self.get_conn()

    def _format_tool_output(self, payload: Any) -> str:
        rendered = str(payload)
        return _truncate_text(rendered, MAX_QUERY_RESULT_CHARS, "SQL RESULT")

    def _init_credential(self):
        """Initialize Azure credential for token-based auth if needed."""
        # Check if connection string uses Azure AD auth that needs token injection
        conn_lower = self.connection_string.lower()
        if "authentication=activedirectory" in conn_lower:
            # Remove the Authentication attribute - we'll use token instead
            import re
            self.connection_string = re.sub(
                r";?\s*Authentication=[^;]+", "", self.connection_string, flags=re.IGNORECASE
            )
            # Also remove Uid if present (not needed for token auth)
            self.connection_string = re.sub(
                r";?\s*Uid=[^;]+", "", self.connection_string, flags=re.IGNORECASE
            )
            # Use AzureCliCredential for WSL (uses az login token)
            self._credential = AzureCliCredential()
            logger.info("Using Azure CLI credential for SQL authentication")

    def get_conn(self):
        try:
            if self._credential:
                # Get token for Azure SQL / Synapse
                # Use the correct scope for Azure Government
                token = self._credential.get_token("https://database.usgovcloudapi.net/.default")
                token_struct = _get_token_struct(token.token)
                conn = pyodbc.connect(self.connection_string, attrs_before={1256: token_struct})
            else:
                conn = pyodbc.connect(self.connection_string)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _execute_query(
        self, query: str, params: Any | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results as a list of dictionaries"""
        logger.debug(f"Executing query: {query}")
        self._last_query_truncated = False
        try:
            with closing(self.get_conn()) as conn:
                with closing(conn.cursor()) as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    if (
                        query.strip()
                        .upper()
                        .startswith(
                            ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")
                        )
                    ):
                        conn.commit()
                        affected = cursor.rowcount
                        logger.debug(f"Write query affected {affected} rows")
                        return [{"affected_rows": affected}]
                    columns = [column[0] for column in cursor.description]
                    rows = cursor.fetchmany(MAX_QUERY_RESULT_ROWS + 1)
                    if len(rows) > MAX_QUERY_RESULT_ROWS:
                        self._last_query_truncated = True
                        rows = rows[:MAX_QUERY_RESULT_ROWS]
                    results = [
                        dict(zip(columns, (_sanitize_cell_value(cell) for cell in row)))
                        for row in rows
                    ]
                    logger.debug(f"Read query returned {len(results)} rows")
                    return results
        except Exception as e:
            logger.error(f"Database error executing query: {e}")
            raise

    def list_tables(self) -> str:
        """List all base tables in the SQL database.

        NOTE: Prefer using list_views() instead of this tool. Views are the primary
        interface for querying data in this database. Only use list_tables if you
        specifically need to see underlying base tables.

        Lists base tables in the Azure SQL (T-SQL) database.
        """

        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        results = self._execute_query(query)
        return self._format_tool_output(results)

    def list_views(self, schema_name: str = "ai") -> str:
        """List all views in the SQL database for the ai schema only.

        *** STEP 1 - START HERE ***
        This is the PRIMARY entry point for discovering available data in the Azure SQL (T-SQL) database.
        ALWAYS use this tool FIRST before describe_table or read_query to understand what views exist.

        Views are the preferred way to access data in this database. They provide:
        - Pre-defined, optimized queries
        - Consistent data access patterns
        - Filtered and transformed data ready for analysis

        After listing views, use describe_table to get column details for a specific view,
        then use read_query to query the view.

        Args:
            schema_name: Ignored. Views are always filtered to schema 'ai'.
        """

        query = (
            "SELECT TABLE_SCHEMA, TABLE_NAME "
            "FROM INFORMATION_SCHEMA.VIEWS "
            "WHERE TABLE_SCHEMA = 'ai'"
        )
        results = self._execute_query(query)
        return self._format_tool_output(results)

    def describe_table(
        self, table_name: str = Field(description="The name of the table to describe")
    ) -> str:
        """Get the schema information for a specific table.

        STEP 2: After using list_tables, you MUST use this to get column names, data types, max lengths,
        nullability, primary keys, and foreign keys for a specific table in the Azure SQL (T-SQL) database.
        This information is essential before executing read_query or write_query. This does not require the database schema name.
        """
        if table_name is None:
            raise ValueError("Missing table_name argument")

        # Get column information
        column_query = """
            SELECT 
                COLUMN_NAME, 
                DATA_TYPE, 
                CHARACTER_MAXIMUM_LENGTH, 
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = ?
        """
        columns = self._execute_query(column_query, (table_name,))

        # Get primary key information
        pk_query = """
            SELECT c.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE c
                ON tc.CONSTRAINT_NAME = c.CONSTRAINT_NAME
            WHERE tc.TABLE_NAME = ? 
                AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        """
        primary_keys = self._execute_query(pk_query, (table_name,))

        # Get foreign key information
        fk_query = """
            SELECT 
                fk.CONSTRAINT_NAME,
                c.COLUMN_NAME,
                c2.TABLE_NAME AS REFERENCED_TABLE,
                c2.COLUMN_NAME AS REFERENCED_COLUMN
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS fk
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE c
                ON fk.CONSTRAINT_NAME = c.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE c2
                ON fk.UNIQUE_CONSTRAINT_NAME = c2.CONSTRAINT_NAME
            WHERE c.TABLE_NAME = ?
        """
        foreign_keys = self._execute_query(fk_query, (table_name,))

        result = (
            "Columns:\n"
            f"{self._format_tool_output(columns)}\n\n"
            "Primary Keys:\n"
            f"{self._format_tool_output(primary_keys)}\n\n"
            "Foreign Keys:\n"
            f"{self._format_tool_output(foreign_keys)}"
        )

        if len(columns) == 0:
            return f"Table '{table_name}' not found or has no columns, ensure you are using the correct table name as defined in the list_tables tool."

        return result

    def _extract_query_objects(self, query: str) -> list[tuple[str | None, str]]:
        """Extract schema/table (or view) references from FROM/JOIN clauses."""
        pattern = re.compile(r"\b(?:FROM|JOIN)\s+([^\s,;]+)", re.IGNORECASE)
        references: list[tuple[str | None, str]] = []
        seen: set[tuple[str | None, str]] = set()

        for raw_token in pattern.findall(query):
            token = raw_token.strip().rstrip(")")
            if not token or token.startswith("("):
                continue

            token = token.split()[0]
            token = token.replace("[", "").replace("]", "").replace('"', "")
            if not token:
                continue

            parts = [part for part in token.split(".") if part]
            if len(parts) >= 2:
                schema_name, object_name = parts[-2], parts[-1]
            else:
                schema_name, object_name = None, parts[0]

            ref = (schema_name, object_name)
            if object_name and ref not in seen:
                seen.add(ref)
                references.append(ref)

        return references

    def _get_object_columns(self, schema_name: str | None, object_name: str) -> list[dict[str, Any]]:
        """Get columns for a referenced table/view from INFORMATION_SCHEMA."""
        if schema_name:
            query = """
                SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """
            return self._execute_query(query, (schema_name, object_name))

        query = """
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY TABLE_SCHEMA, ORDINAL_POSITION
        """
        return self._execute_query(query, (object_name,))

    def _build_schema_feedback_for_query(self, query: str) -> str:
        """Build schema context for objects referenced in the failed SQL query."""
        references = self._extract_query_objects(query)
        if not references:
            return "Could not infer table/view names from the query. Use list_views() and describe_table() first."

        payload: list[dict[str, Any]] = []
        for schema_name, object_name in references:
            full_name = f"{schema_name}.{object_name}" if schema_name else object_name
            try:
                columns = self._get_object_columns(schema_name, object_name)
            except Exception as schema_error:
                payload.append(
                    {
                        "object": full_name,
                        "error": f"Failed to load schema: {str(schema_error)}",
                        "columns": [],
                    }
                )
                continue

            payload.append(
                {
                    "object": full_name,
                    "columns": columns,
                }
            )

        return self._format_tool_output(payload)

    def read_query(
        self, query: str = Field(description="T-SQL SELECT query to execute")
    ) -> str:
        """Execute a SELECT query on the SQL database.

        REQUIRED WORKFLOW:
        STEP 3a: First, query distinct values for any columns you plan to filter on.
                 Example: SELECT DISTINCT status FROM schema.orders
                 Example: SELECT DISTINCT category FROM alternative_schema.products
                 This shows you what values actually exist in the database.

        STEP 3b: After verifying actual values exist, construct your filtered query.
                 Use ONLY the exact values you found in step 3a for WHERE clauses.

        DO NOT skip step 3a. DO NOT assume what values exist. DO NOT filter without first checking.
        Executes T-SQL SELECT statements on the Azure SQL database.
        """

        # Normalize the query for checking
        normalized_query = query.strip().upper()

        # Only allow SELECT statements
        if not normalized_query.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed for read_query")

        # Check if this is a filtered/aggregation query without having checked distinct values first
        has_where = "WHERE" in normalized_query
        has_count = "COUNT(" in normalized_query
        has_sum = "SUM(" in normalized_query
        has_avg = "AVG(" in normalized_query
        has_group_by = "GROUP BY" in normalized_query

        # Block dangerous keywords that could modify data or schema
        dangerous_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "MERGE",
            "EXEC",
            "EXECUTE",
            "CALL",
            "GRANT",
            "REVOKE",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
        ]

        for keyword in dangerous_keywords:
            if keyword in normalized_query:
                raise ValueError(
                    f"Keyword '{keyword}' is not allowed in read-only queries"
                )

        try:
            results = self._execute_query(query)
        except Exception as e:
            schema_feedback = self._build_schema_feedback_for_query(query)
            return (
                "Query failed to execute.\n"
                f"Database error: {str(e)}\n\n"
                "Schema for referenced objects:\n"
                f"{schema_feedback}\n\n"
                "Retry the query using only the table/view and column names shown above."
            )

        rendered_results = self._format_tool_output(results)
        truncated_note = (
            "\n\nNOTE: Query results were truncated to keep tool output within context limits."
            if self._last_query_truncated
            else ""
        )

        if has_where or has_count or has_sum or has_avg or has_group_by:
            return (
                f"Your results are: {rendered_results}{truncated_note}\n"
                "NOTE: You are attempting to filter or aggregate data"
                "Although this query completed, please make sure you have:\n"
                "1. Run SELECT DISTINCT queries on any columns you filter on\n"
                "2. Verify the actual values that exist in the database\n"
                "3. Then construct your filtered/aggregated query using only those verified values\n\n"
                "If you have not performed these steps, you will need to reattempt your query. "
                "Do NOT proceed without checking distinct values first."
            )
        else:
            return f"{rendered_results}{truncated_note}"


# # from sqlite_db import SqliteDatabase
# db = SqlDatabase(connection_string)
