import sqlite3
conn = sqlite3.connect(r'E:\DATA\gx498\Desktop\course work\open_deep_research\test.db')
cur = conn.cursor()

print('=== 简单查询类 ===')
print('Q6: units总和')
print(cur.execute('SELECT SUM(units) FROM sales').fetchone())

print('Q7: East大区的记录数')
print(cur.execute("SELECT COUNT(*) FROM sales WHERE region='East'").fetchone())

print('Q8: revenue最大值对应的product_line')
print(cur.execute('SELECT product_line, MAX(revenue) FROM sales').fetchone())

print('Q9: revenue最小值')
print(cur.execute('SELECT ROUND(MIN(revenue),2) FROM sales').fetchone())

print('=== 跨表关联类 ===')
print('Q10: 按category分组统计总units')
for row in cur.execute('''
    SELECT p.category, SUM(s.units)
    FROM sales s JOIN products p ON s.product_line=p.product_line
    GROUP BY p.category ORDER BY p.category
''').fetchall():
    print(row)

print('Q11: launch_year>2021的产品线总销售额')
for row in cur.execute('''
    SELECT s.product_line, ROUND(SUM(s.revenue),2)
    FROM sales s JOIN products p ON s.product_line=p.product_line
    WHERE p.launch_year > 2021
    GROUP BY s.product_line ORDER BY s.product_line
''').fetchall():
    print(row)

print('Q12: 每个产品线的平均每笔revenue(ROUND到2位)')
for row in cur.execute('''
    SELECT product_line, ROUND(AVG(revenue),2)
    FROM sales GROUP BY product_line ORDER BY product_line
''').fetchall():
    print(row)

conn.close()