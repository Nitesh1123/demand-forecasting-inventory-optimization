import os
import sqlite3
from pathlib import Path

import pandas as pd

SQL_PATH = "sql"
DATABASE_PATH = "outputs/powerbi/demand.db"
OUTPUT_PATH = "outputs/sql_results"

output_path = os.path.abspath(OUTPUT_PATH)
os.makedirs(output_path, exist_ok=True)

with sqlite3.connect(os.path.abspath(DATABASE_PATH)) as connection:
    for query_path in sorted(Path(SQL_PATH).glob("*.sql")):
        result = pd.read_sql_query(query_path.read_text(encoding="utf-8"), connection)
        result.to_csv(os.path.join(output_path, f"{query_path.stem}.csv"), index=False)
        print(f"\n== {query_path.stem} ({len(result)} rows)")
        print(result.head(10).to_string(index=False))
