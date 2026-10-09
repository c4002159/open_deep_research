"""检索基准集（不依赖 LLM 评分）。

gold 为 (文件名, 章节序号) 列表，命中任意一个即算该 gold 被召回。
query 刻意用口语化改写，不照抄文档原句，避免只考关键词匹配。
章节序号 = 文档里 "## N." 的 N。
"""

_C = "salesCommissionRules.md"
_N = "DataFieldNamingRules.md"
_R = "refundPolicy.md"
_P = "dataPermission.md"
_M = "reportMetrics.md"
_T = "customerTier.md"
_O = "regionOwners.md"
_G = "productCatalog.md"
_K = "knowledgeBaseUsage.md"

RETRIEVAL_BENCH_CASES = [
    # --- 原有 7 条（补上 gold 标注）---
    {"id": "recall_commission_rate", "query": "基础产品的提成比例是多少", "gold": [(_C, 2)]},
    {"id": "recall_strategic_rate", "query": "战略产品的提成比例是多少", "gold": [(_C, 2)]},
    {"id": "recall_large_order_bonus", "query": "大单激励的条件和比例是什么", "gold": [(_C, 2), (_T, 3)]},
    {"id": "recall_bonus_period", "query": "提成发放周期是多久", "gold": [(_C, 3)]},
    {"id": "recall_amount_field_naming", "query": "订单金额字段命名规范", "gold": [(_N, 2)]},
    {"id": "recall_quantity_field_naming", "query": "数量字段应该怎么命名", "gold": [(_N, 2)]},
    {"id": "recall_cross_doc", "query": "提成计算里用到的销售额字段命名规范是什么", "gold": [(_C, 2), (_N, 2)]},
    # --- 新增：口语化 / 改写 ---
    {"id": "refund_window", "query": "客户收到货三十天以后还能退钱吗", "gold": [(_R, 1)]},
    {"id": "refund_approver", "query": "退款三万块要找谁批", "gold": [(_R, 2)]},
    {"id": "refund_commission", "query": "客户退了款，销售的提成怎么处理", "gold": [(_R, 3), (_C, 3)]},
    {"id": "refund_rejected", "query": "退款申请被驳回的话提成会受影响吗", "gold": [(_R, 3)]},
    {"id": "refund_cfo_deadline", "query": "超过五万的退款多久之内要处理完", "gold": [(_R, 2)]},
    {"id": "sla_p1", "query": "服务完全不可用时多久要响应", "gold": [(_R, 4)]},
    {"id": "data_levels", "query": "哪些数据属于机密级别", "gold": [(_P, 1)]},
    {"id": "query_row_limit", "query": "一次查询最多能返回多少行数据", "gold": [(_P, 2)]},
    {"id": "mask_phone", "query": "客户手机号展示的时候怎么打码", "gold": [(_P, 3)]},
    {"id": "salary_in_report", "query": "员工薪资能放进对外分享的报告吗", "gold": [(_P, 3)]},
    {"id": "leak_penalty", "query": "把数据泄露出去会怎么处罚", "gold": [(_P, 4)]},
    {"id": "revenue_tax", "query": "销售额是含税的还是不含税", "gold": [(_M, 1)]},
    {"id": "units_split", "query": "同一订单被拆成多条记录，销量怎么算", "gold": [(_M, 2)]},
    {"id": "avg_ticket", "query": "客单价怎么计算", "gold": [(_M, 3)]},
    {"id": "q2_months", "query": "二季度是几月到几月", "gold": [(_M, 4)]},
    {"id": "yoy_def", "query": "同比增长是跟什么时候比", "gold": [(_M, 4)]},
    {"id": "strategic_customer", "query": "年度消费超过五十万算什么级别的客户", "gold": [(_T, 1)]},
    {"id": "key_account_discount", "query": "给重点客户最多能打多少折扣", "gold": [(_T, 2)]},
    {"id": "strategic_discount", "query": "战略客户的折扣上限是多少", "gold": [(_T, 2)]},
    {"id": "big_order_review", "query": "大单需要提前几天提交评审", "gold": [(_T, 3)]},
    {"id": "region_list", "query": "公司一共有哪几个销售大区", "gold": [(_O, 1)]},
    {"id": "region_manager_report", "query": "区域经理每个月要交什么材料", "gold": [(_O, 2)]},
    {"id": "region_kpi_weight", "query": "区域考核看哪些指标，各占多大权重", "gold": [(_O, 3)]},
    {"id": "new_customer_weight", "query": "新客户数量在考核里占多少比重", "gold": [(_O, 3)]},
    {"id": "product_category", "query": "Data Platform 属于哪一类产品", "gold": [(_G, 1), (_G, 2)]},
    {"id": "ai_tools_launch", "query": "AI Tools 是哪一年上市的", "gold": [(_G, 1)]},
    {"id": "new_product_def", "query": "什么样的产品算新品", "gold": [(_G, 3)]},
    {"id": "delist_rule", "query": "哪些产品会被拉去做退市评估", "gold": [(_G, 3)]},
    {"id": "assistant_unknown", "query": "助手碰到制度里没写的问题应该怎么回答", "gold": [(_K, 2)]},
    {"id": "sql_retry_limit", "query": "SQL 连续报错几次之后要告诉用户", "gold": [(_K, 3)]},
    {"id": "assistant_modify", "query": "能不能让助手帮我改数据库里的数据", "gold": [(_K, 2), (_P, 2)]},
    {"id": "naming_style", "query": "字段命名是用驼峰还是下划线", "gold": [(_N, 1)]},
    {"id": "product_key_field", "query": "产品的唯一标识字段叫什么名字", "gold": [(_N, 2)]},
    {"id": "security_suite_rate", "query": "Security Suite 的提成按百分之几算", "gold": [(_C, 2), (_G, 2)]},
]
