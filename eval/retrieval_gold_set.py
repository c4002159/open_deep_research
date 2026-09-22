RETRIEVAL_GOLD_CASES = [
    {
        "id": "recall_commission_rate",
        "query": "基础产品的提成比例是多少",
        "gold_answer": "基础产品（如AI Tools、Security Suite）按单笔订单净销售额（revenue）的5%计提提成。",
    },
    {
        "id": "recall_strategic_rate",
        "query": "战略产品的提成比例是多少",
        "gold_answer": "战略产品（如Data Platform、Cloud Services）按单笔订单净销售额（revenue）的8%计提提成。",
    },
    {
        "id": "recall_large_order_bonus",
        "query": "大单激励的条件和比例是什么",
        "gold_answer": "若单笔订单sales units超过100，该笔订单额外享受2%的大单激励。",
    },
    {
        "id": "recall_bonus_period",
        "query": "提成发放周期是多久",
        "gold_answer": "提成每季度核算一次，并在次月随工资一同发放。若订单发生退款，需在下个季度的提成中进行等额扣除。",
    },
    {
        "id": "recall_amount_field_naming",
        "query": "订单金额字段命名规范",
        "gold_answer": "订单金额相关字段统一命名为revenue，严禁使用price或amount。",
    },
    {
        "id": "recall_quantity_field_naming",
        "query": "数量字段应该怎么命名",
        "gold_answer": "商品件数统一命名为units，严禁使用qty或count。",
    },
    {
        "id": "recall_cross_doc",
        "query": "提成计算里用到的销售额字段命名规范是什么",
        "gold_answer": "提成计算基于revenue字段（单笔订单净销售额）；该字段的命名规范中，订单金额统一命名为revenue，严禁使用price或amount。",
    },
]

# 单独放一类：知识库里没有对应内容的查询。
# ContextualRecallMetric要求有一个明确的gold_answer才能打分，这类"该查不到"的场景
# 不适合走这套指标（打分逻辑本身假设"标准答案是存在且该被找到的"），单独用之前那份
# 简单脚本里"检查有没有诚实拒答"的逻辑去测，不要混进这个文件里评分。
OUT_OF_SCOPE_QUERIES = [
    "公司年假政策",
]