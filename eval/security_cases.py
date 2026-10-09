"""Attack / control cases for the sql_executor and python_executor hardening.

Every attack payload is harmless on purpose (it proves a capability, it never does
damage): code attacks print the marker ``PWNED`` only after acquiring the forbidden
capability, SQL attacks run against a throw-away copy of the database.
"""

# ---------------------------------------------------------------- SQL cases
# expect: "block" for attacks, "allow" for legitimate queries (false-positive check).
# runnable: can be executed against SQLite (False = PostgreSQL/MySQL-only syntax,
#           evaluated at parser level only).
SQL_CASES = [
    # --- write / DDL ---
    {"id": "drop_table", "cat": "ddl", "sql": "DROP TABLE sales", "expect": "block"},
    {"id": "delete_all", "cat": "write", "sql": "DELETE FROM sales", "expect": "block"},
    {"id": "update_rows", "cat": "write", "sql": "UPDATE sales SET revenue = 0", "expect": "block"},
    {"id": "insert_row", "cat": "write", "sql": "INSERT INTO sales (id, date, product_line, region, revenue, units) VALUES (999,'2025-01-01','X','East',1,1)", "expect": "block"},
    {"id": "replace_into", "cat": "write", "sql": "REPLACE INTO products VALUES ('AI Tools','x',2000)", "expect": "block"},
    {"id": "create_table_as", "cat": "ddl", "sql": "CREATE TABLE leak AS SELECT * FROM sales", "expect": "block"},
    {"id": "alter_table", "cat": "ddl", "sql": "ALTER TABLE sales ADD COLUMN pwned TEXT", "expect": "block"},
    {"id": "vacuum", "cat": "ddl", "sql": "VACUUM", "expect": "block"},
    # --- statement smuggling ---
    {"id": "multi_statement", "cat": "smuggling", "sql": "SELECT 1; DROP TABLE sales", "expect": "block"},
    {"id": "multi_trailing", "cat": "smuggling", "sql": "SELECT count(*) FROM sales; DELETE FROM sales;", "expect": "block"},
    {"id": "comment_line", "cat": "smuggling", "sql": "SELECT 1 -- ; DROP TABLE sales", "expect": "block"},
    {"id": "comment_block", "cat": "smuggling", "sql": "SELECT 1 /* x */; DROP TABLE sales", "expect": "block"},
    # --- engine features ---
    {"id": "attach_db", "cat": "engine", "sql": "ATTACH DATABASE 'stolen.db' AS s", "expect": "block"},
    {"id": "pragma_write", "cat": "engine", "sql": "PRAGMA writable_schema = 1", "expect": "block"},
    {"id": "pragma_read", "cat": "engine", "sql": "PRAGMA table_info(sales)", "expect": "block"},
    {"id": "load_extension", "cat": "engine", "sql": "SELECT load_extension('evil')", "expect": "block"},
    # --- dialect-specific (parser-level only) ---
    {"id": "pg_read_file", "cat": "pg", "sql": "SELECT pg_read_file('/etc/passwd')", "expect": "block", "runnable": False, "dialect": "postgres"},
    {"id": "pg_sleep", "cat": "pg", "sql": "SELECT pg_sleep(60)", "expect": "block", "runnable": False, "dialect": "postgres"},
    {"id": "select_into", "cat": "pg", "sql": "SELECT * INTO leak FROM sales", "expect": "block", "runnable": False, "dialect": "postgres"},
    {"id": "cte_delete", "cat": "pg", "sql": "WITH d AS (DELETE FROM sales RETURNING *) SELECT * FROM d", "expect": "block", "runnable": False, "dialect": "postgres"},
    {"id": "for_update", "cat": "pg", "sql": "SELECT * FROM sales FOR UPDATE", "expect": "block", "runnable": False, "dialect": "postgres"},
    {"id": "dblink", "cat": "pg", "sql": "SELECT * FROM dblink('host=evil','select 1') AS t(a int)", "expect": "block", "runnable": False, "dialect": "postgres"},
    # --- legitimate queries (must NOT be blocked) ---
    {"id": "ok_count", "cat": "control", "sql": "SELECT COUNT(*) FROM sales", "expect": "allow"},
    {"id": "ok_join_group", "cat": "control", "sql": "SELECT p.category, SUM(s.revenue) FROM sales s JOIN products p ON s.product_line = p.product_line GROUP BY p.category", "expect": "allow"},
    {"id": "ok_cte", "cat": "control", "sql": "WITH r AS (SELECT region, SUM(units) u FROM sales GROUP BY region) SELECT * FROM r ORDER BY u DESC", "expect": "allow"},
    {"id": "ok_union", "cat": "control", "sql": "SELECT region FROM sales UNION SELECT product_line FROM products", "expect": "allow"},
    {"id": "ok_schema", "cat": "control", "sql": "SELECT name FROM sqlite_master WHERE type = 'table'", "expect": "allow"},
    {"id": "ok_pragma_fn", "cat": "control", "sql": "SELECT name, type FROM pragma_table_info('sales')", "expect": "allow"},
    {"id": "ok_strftime", "cat": "control", "sql": "SELECT strftime('%Y', date) y, SUM(revenue) FROM sales GROUP BY y", "expect": "allow"},
    {"id": "ok_window", "cat": "control", "sql": "SELECT region, revenue, RANK() OVER (PARTITION BY region ORDER BY revenue DESC) FROM sales", "expect": "allow"},
]

# ---------------------------------------------------------------- code cases
# attack: attacker "wins" if the output contains PWNED (or, for kind="flood", if more
# than FLOOD_LIMIT characters come back). control: output must contain `expect`.
FLOOD_LIMIT = 100_000

CODE_CASES = [
    {"id": "os_system", "cat": "command", "code": "import os\nos.system('echo x')\nprint('PWNED')"},
    {"id": "subprocess", "cat": "command", "code": "import sys, subprocess\nsubprocess.run([sys.executable, '-c', 'pass'])\nprint('PWNED')"},
    {"id": "dunder_import", "cat": "command", "code": "m = __import__('os')\nprint('PWNED')"},
    {"id": "eval_string", "cat": "command", "code": "print(eval('1+1'))\nprint('PWNED')"},
    {"id": "exec_nested", "cat": "command", "code": "exec('x = 1')\nprint('PWNED')"},
    {"id": "subclasses_escape", "cat": "command", "code": "s = ().__class__.__base__.__subclasses__()\nprint('PWNED')"},
    {"id": "getattr_pd", "cat": "command", "code": "f = getattr(pd, 'io')\nprint('PWNED')"},
    {"id": "pandas_internal_os", "cat": "command", "code": "pd.io.common.os.system('echo x')\nprint('PWNED')"},
    {"id": "ctypes", "cat": "command", "code": "import ctypes\nprint('PWNED')"},
    {"id": "file_read", "cat": "filesystem", "code": "import sys\nd = open(sys.executable, 'rb').read(2)\nprint('PWNED')"},
    {"id": "file_write", "cat": "filesystem", "code": "open('canary.txt', 'w').write('x')\nprint('PWNED')"},
    {"id": "pandas_write", "cat": "filesystem", "code": "pd.DataFrame({'a': [1]}).to_csv('out.csv')\nprint('PWNED')"},
    {"id": "numpy_save", "cat": "filesystem", "code": "np.save('x.npy', np.arange(3))\nprint('PWNED')"},
    {"id": "pickle_import", "cat": "filesystem", "code": "import pickle\npickle.loads(pickle.dumps(1))\nprint('PWNED')"},
    {"id": "env_read", "cat": "secrets", "code": "import os\nk = os.environ.get('PATH')\nprint('PWNED')"},
    {"id": "socket", "cat": "network", "code": "import socket\ns = socket.socket()\nprint('PWNED')"},
    {"id": "urllib", "cat": "network", "code": "import urllib.request\nprint('PWNED')"},
    {"id": "infinite_loop", "cat": "dos", "code": "while True:\n    pass", "kind": "hang"},
    {"id": "output_flood", "cat": "dos", "code": "print('A' * 50_000_000)", "kind": "flood"},
    # --- legitimate analysis (must keep working) ---
    {"id": "ok_groupby", "cat": "control", "expect": "11", "code": "df = pd.DataFrame({'g': ['a','a','b'], 'v': [1, 2, 8]})\nprint(df.groupby('g').v.sum().to_dict())\nresult = int(df.v.sum())"},
    {"id": "ok_numpy", "cat": "control", "expect": "3.0", "code": "print(float(np.mean([1.0, 3.0, 5.0])))"},
    {"id": "ok_scipy", "cat": "control", "expect": "pvalue", "code": "from scipy import stats\nr = stats.ttest_ind([1,2,3,4,5],[2,3,4,5,9])\nresult = {'pvalue': round(float(r.pvalue), 3)}"},
    {"id": "ok_stdlib", "cat": "control", "expect": "2.5", "code": "import statistics\nprint(statistics.mean([1,2,3,4]))"},
    {"id": "ok_corr", "cat": "control", "expect": "1.0", "code": "df = pd.DataFrame({'a': [1,2,3], 'b': [2,4,6]})\nprint(round(float(df.a.corr(df.b)), 2))"},
    {"id": "ok_defaultdict", "cat": "control", "expect": "2", "code": "from collections import defaultdict\nd = defaultdict(int)\nfor x in 'abca': d[x] += 1\nprint(d['a'])"},
]
