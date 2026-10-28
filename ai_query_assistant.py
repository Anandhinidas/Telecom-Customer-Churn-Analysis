import os
import re
import mysql.connector
from mysql.connector import Error
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# STEP 1: Connect to MySQL and run a query
# ============================================================

def run_sql_query(query: str) -> dict:
    """
    Connects to the telecom_churn_db MySQL database, runs a SELECT query,
    and returns the results as a list of dicts (column_name: value).
    """
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',                     
            password='MYSQL_PASSWORD',   
            database='telecom_churn_db'
        )

        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()

        return {"success": True, "data": results}

    except Error as e:
        return {"success": False, "error": str(e)}

    finally:
        if connection and connection.is_connected():
            connection.close()


# ============================================================
# STEP 2: Safety check + tool definition for Gemini
# ============================================================

def is_safe_select_query(query: str) -> bool:
    """
    Real code-level safety check — only allows SELECT queries,
    blocks anything that could modify data.
    """
    query_clean = query.strip().upper()
    if not query_clean.startswith("SELECT"):
        return False
    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT"]
    return not any(re.search(rf"\b{kw}\b", query_clean) for kw in dangerous_keywords)


sql_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="run_sql_query",
        description=(
            "Runs a read-only SQL SELECT query against the telecom_churn_db MySQL database "
            "to answer questions about customer churn. Use this whenever the user asks a "
            "question that requires looking up or calculating something from the customer_churn "
            "table (e.g., counts, averages, churn rates, breakdowns by category). "
            "Only SELECT statements are allowed — never write INSERT, UPDATE, DELETE, or DROP."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid MySQL SELECT query to run against the customer_churn table."
                }
            },
            "required": ["query"]
        }
    )
])


# ============================================================
# STEP 3: System prompt — schema + validated findings
# ============================================================

system_prompt = """You are a Data Analyst Assistant for a Telecom Customer Churn Analysis project. 
You answer business questions about customer churn using real data from a MySQL database, 
combined with pre-validated findings from prior statistical analysis and machine learning.

DATABASE SCHEMA
Table: customer_churn (7,043 rows)
Key columns:
- Contract: 'Month-to-month', 'One year', 'Two year'
- tenure: integer, months as a customer
- PaymentMethod: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
- MonthlyCharges: decimal
- TotalCharges: decimal
- InternetService: 'DSL', 'Fiber optic', 'No'
- TechSupport: 'Yes', 'No', 'No internet service'
- OnlineSecurity: 'Yes', 'No', 'No internet service'
- Churn: 'Yes' or 'No'

VALIDATED FINDINGS (from prior SQL, statistical testing, and Logistic Regression analysis — 
use these as ground truth when relevant, don't recompute unless the user asks a new question 
these findings don't cover):
- Overall churn rate: 26.54%
- Churn by contract: Month-to-month 42.71%, One year 11.27%, Two year 2.83% 
  (chi-square test: chi2=1184.60, p<0.001 — statistically significant)
- Churn by tenure bucket: 0-12mo 47.44%, 13-24mo 28.71%, 25-48mo 20.39%, 49+mo 9.51%
- Churn by payment method: Electronic check 45.29%, Mailed check 19.11%, 
  Bank transfer 16.71%, Credit card 15.24%
- Churned customers: avg MonthlyCharges $74.44 vs $61.27 retained (t=18.41, p<0.001); 
  avg tenure 18.0 vs 37.6 months
- Top churn drivers (Logistic Regression coefficients): Two-year contract (-1.39), 
  tenure (-1.30), Fiber optic internet (+0.95), TotalCharges (+0.62), TechSupport (-0.35), 
  Electronic check (+0.42), PaperlessBilling (+0.34)

INSTRUCTIONS
- When a user asks a question that requires new data (not covered by the findings above), 
  use the run_sql_query tool to query the database directly.
- Only use SELECT statements. Never attempt to modify data.
- When you already have the answer from the validated findings above, you may answer directly 
  without querying, but you can still query to double-check or drill deeper if the user asks 
  a follow-up.
- Always answer in plain business language, not raw SQL output — explain what the numbers mean.
- If a query returns no data or an error, say so honestly rather than guessing.
"""


# ============================================================
# STEP 4: The request-response loop (using google-genai)
# ============================================================

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"   # current, supported model name

def ask_churn_assistant(user_question: str) -> str:
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[sql_tool]
        )
    )

    response = chat.send_message(user_question)
    part = response.candidates[0].content.parts[0]

    if part.function_call and part.function_call.name == "run_sql_query":
        query = part.function_call.args["query"]
        print(f"[Gemini wants to run]: {query}")

        if not is_safe_select_query(query):
            tool_output = {"success": False, "error": "Only SELECT queries are permitted."}
        else:
            tool_output = run_sql_query(query)

        # Send the tool result back through the SAME chat session —
        # the chat object handles thought_signature and history automatically
        follow_up = chat.send_message(
            types.Part.from_function_response(
                name="run_sql_query",
                response={"result": str(tool_output)}
            )
        )
        return follow_up.text

    return response.text


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    answer = ask_churn_assistant("What's the churn rate for customers on fiber optic internet?")
    print("\nFinal answer:")
    print(answer)