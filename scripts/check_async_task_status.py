import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
job_id = sys.argv[1] if len(sys.argv) > 1 else ""
connection = pymysql.connect(
    host=os.getenv("MYSQL_HOST", "localhost"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE", "qilin_stock"),
)
try:
    with connection.cursor() as cursor:
        cursor.execute("SELECT status FROM async_tasks WHERE job_id=%s", (job_id,))
        row = cursor.fetchone()
    print(row[0] if row else "missing")
finally:
    connection.close()
