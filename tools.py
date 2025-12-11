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
        """List all base tables in the SQL database.

        NOTE: Prefer using list_views() instead of this tool. Views are the primary
        interface for querying data in this database. Only use list_tables if you
        specifically need to see underlying base tables.

        Lists base tables in the Azure SQL (T-SQL) database.
        """

        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        results = self._execute_query(query)
        return str(results)

    def list_views(self, schema_name: str = "dbo") -> str:
        """List all views in the SQL database for a specific schema.

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
            schema_name: The schema to filter views by (default: 'dbo')
        """

        query = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS'"
        results = self._execute_query(query)
        return str(results)

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

        if len(columns) == 0:
            return f"Table '{table_name}' not found or has no columns, ensure you are using the correct table name as defined in the list_tables tool."

        return result

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

        # Execute the query
        results = self._execute_query(query)

        if has_where or has_count or has_sum or has_avg or has_group_by:
            return (
                f"Your results are: {str(results)}\n"
                "NOTE: You are attempting to filter or aggregate data"
                "Although this query completed, please make sure you have:\n"
                "1. Run SELECT DISTINCT queries on any columns you filter on\n"
                "2. Verify the actual values that exist in the database\n"
                "3. Then construct your filtered/aggregated query using only those verified values\n\n"
                "If you have not performed these steps, you will need to reattempt your query. "
                "Do NOT proceed without checking distinct values first."
            )
        else:
            return str(results)


# # from sqlite_db import SqliteDatabase
# db = SqlDatabase(connection_string)
