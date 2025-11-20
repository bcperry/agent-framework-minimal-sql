import logging

from pydantic import Field
import pyodbc
from contextlib import closing
from typing import Any


# # Load environment variables from .env file (for local development)
# load_dotenv()

# # Get connection string from environment
# connection_string = os.environ.get("AZURE_SQL_CONNECTIONSTRING")
# print(f"Connected using: {connection_string}")
# if not connection_string:
#     raise ValueError("AZURE_SQL_CONNECTIONSTRING environment variable is required")


logger = logging.getLogger("tools")


class SqlDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.get_conn()

    def get_conn(self):
        try:
            conn = pyodbc.connect(self.connection_string)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results as a list of dictionaries"""
        logger.debug(f"Executing query: {query}")
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
                    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    logger.debug(f"Read query returned {len(results)} rows")
                    return results
        except Exception as e:
            logger.error(f"Database error executing query: {e}")
            raise

    def list_tables(self) -> str:
        """List all tables in the SQL database.

        STEP 1: Start here to discover available tables in the Azure SQL (T-SQL) database.
        Use this before describe_table or read_query to understand what tables exist.
        """

        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        results = self._execute_query(query)
        return str(results)

    def describe_table(
        self, table_name: str = Field(description="Name of the table to describe")
    ) -> str:
        """Get the schema information for a specific table.

        STEP 2: After using list_tables, use this to get column names, data types, max lengths,
        nullability, primary keys, and foreign keys for a specific table in the Azure SQL (T-SQL) database.
        This information is essential before executing read_query or write_query.
        """
        if table_name is None:
            raise ValueError("Missing table_name argument")

        # Get column information
        column_query = f"""
            SELECT 
                COLUMN_NAME, 
                DATA_TYPE, 
                CHARACTER_MAXIMUM_LENGTH, 
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}'
        """
        columns = self._execute_query(column_query)

        # Get primary key information
        pk_query = f"""
            SELECT c.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE c
                ON tc.CONSTRAINT_NAME = c.CONSTRAINT_NAME
            WHERE tc.TABLE_NAME = '{table_name}' 
                AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        """
        primary_keys = self._execute_query(pk_query)

        # Get foreign key information
        fk_query = f"""
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
            WHERE c.TABLE_NAME = '{table_name}'
        """
        foreign_keys = self._execute_query(fk_query)

        result = f"Columns:\n{columns}\n\nPrimary Keys:\n{primary_keys}\n\nForeign Keys:\n{foreign_keys}"
        return result

    def read_query(
        self, query: str = Field(description="T-SQL SELECT query to execute")
    ) -> str:
        """Execute a SELECT query on the SQL database.

        STEP 3: After understanding table schemas with describe_table, use this to query data.
        Executes T-SQL SELECT statements on the Azure SQL database.
        Use table schemas from describe_table to construct accurate queries with correct column names and types.

        CRITICAL: Before writing any WHERE clause or filtering condition, you MUST execute
        "SELECT DISTINCT column_name FROM table_name" queries to see what values actually exist
        in the database. This is NOT optional. Always verify the actual data values before
        constructing filters to avoid empty results or incorrect assumptions about the data.
        """

        # Normalize the query for checking
        normalized_query = query.strip().upper()

        # Only allow SELECT statements
        if not normalized_query.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed for read_query")

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

        # Execute the query
        results = self._execute_query(query)
        return str(results)


# # from sqlite_db import SqliteDatabase
# db = SqlDatabase(connection_string)
