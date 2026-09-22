EVAL_CASES = [
    # ==========================================
    # 第一类：简单查询类 (Simple Queries)
    # ==========================================
    {
        "id": "simple_count",
        "question": "sales表总共有多少条记录？不用问我连接信息，直接用工具查。",
        "expected_output": "80",
    },
    {
        "id": "total_revenue",
        "question": "sales表的全部历史总销售额是多少？不用问我连接信息，直接用工具查。",
        "expected_output": "2345328.08",
    },
    {
        "id": "sum_units",
        "question": "sales表中全部的销量(units)总和是多少？不用问我连接信息，直接用工具查。",
        "expected_output": "7663",
    },
    {
        "id": "count_east_region",
        "question": "East 大区总共有多少条销售记录？不用问我连接信息，直接用工具查。",
        "expected_output": "16",
    },
    {
        "id": "max_revenue_product",
        "question": "单笔销售额(revenue)最高的是哪个产品线，销售额是多少？不用问我连接信息，直接用工具查。",
        "expected_output_db_available": "Data Platform|49879.66",
        "match_mode_db_available": "all",
        "expected_output_db_unavailable": "无法|未配置|internal_db_url|没有可用|基础设施错误|失败",
        "match_mode_db_unavailable": "any",
        "expected_output": "Data Platform|49879.66" # 兜底
    },
    {
        "id": "min_revenue",
        "question": "单笔销售额(revenue)的最小值是多少？不用问我连接信息，直接用工具查。",
        "expected_output": "5146.04",
    },

    # ==========================================
    # 第二类：跨表关联/分组类 (JOINs & Group By)
    # ==========================================
    {
        "id": "groupby_product_revenue",
        "question": "按产品线分组统计总销售额，并标注每个产品线属于哪个category。不用问我连接信息，直接用工具查。",
        "expected_output": "518489.55|567270.37|826968.89|432599.27",
    },
    {
        "id": "groupby_region_units",
        "question": "按大区（region）分组统计总销量（units）。不用问我连接信息，直接用工具查。",
        "expected_output_db_available": "1623|2274|1881|1885",
        "match_mode_db_available": "all",
        "expected_output_db_unavailable": "无法|未配置|internal_db_url|没有可用|基础设施错误|失败",
        "match_mode_db_unavailable": "any",
        "expected_output": "1623|2274|1881|1885"
    },
    {
        "id": "groupby_category_units",
        "question": "按 category 分组统计总销量(units)。注意 category 在 products 表里。不用问我连接信息，直接用工具查。",
        "expected_output_db_available": "1686|2304|2404|1269",
        "match_mode_db_available": "all",
        "expected_output_db_unavailable": "无法|未配置|internal_db_url|没有可用|基础设施错误|失败",
        "match_mode_db_unavailable": "any",
        "expected_output": "1686|2304|2404|1269"
    },
    {
        "id": "revenue_launch_after_2021",
        "question": "launch_year 在 2021 年之后的产品线的总销售额是多少？不用问我连接信息，直接用工具查。",
        "expected_output_db_available": "AI Tools|518489.55",
        "match_mode_db_available": "all",
        "expected_output_db_unavailable": "无法|未配置|internal_db_url|没有可用|基础设施错误|失败",
        "match_mode_db_unavailable": "any",
        "expected_output": "AI Tools|518489.55"
    },
    {
        "id": "avg_revenue_per_product",
        "question": "计算每个产品线的平均每笔 revenue（结果保留2位小数）。不用问我连接信息，直接用工具查。",
        "expected_output": "28804.97|28363.52|29534.6|30899.95",
    },

    # ==========================================
    # 第三类：故意报错/异常恢复类 (Error Recovery)
    # 这一类的 match_mode 全都是 "any"，只要意思对了就算过
    # ==========================================
    {
        "id": "typo_column_recovery",
        "question": "帮我查一下 sales 表里的 order_status 字段有哪些取值。不用问我连接信息，直接用工具查。",
        "expected_output": "不存在|没有|does not exist|无此字段",
        "match_mode": "any",
    },
    {
        "id": "typo_table_name",
        "question": "帮我查一下 orders 表里的总销售额。不用问我连接信息，直接用工具查。",
        "expected_output": "不存在|没有|no such table|无此表",
        "match_mode": "any",
    },
    {
        "id": "wrong_enum_value",
        "question": "帮我查一下 region 为 'Center' 的总销量。不用问我连接信息，直接用工具查。",
        "expected_output": "0|不存在|没有记录|无记录|空白",
        "match_mode": "any",
    },
    {
        "id": "wrong_join_key",
        "question": "帮我查一下 product_id = 1 的产品的总销量。不用问我连接信息，直接用工具查。",
        "expected_output": "不存在|无此字段|没有这个字段|product_line",
        "match_mode": "any",
    },
]