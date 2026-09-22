# 研发中心数据表字段命名红线规范

## 1. 核心原则
- 所有数据库表名、字段名必须使用小写字母。
- 单词之间必须使用下划线（_）分割，即 snake_case。
- 严禁使用驼峰命名法（CamelCase）或拼音。

## 2. 常见业务对象标准命名
- 产品唯一标识：统一使用 `product_line`，严禁使用 `product_id` 或 `productCode`。
- 金额相关：订单金额统一命名为 `revenue`，严禁使用 `price` 或 `amount`。
- 数量相关：商品件数统一命名为 `units`，严禁使用 `qty` 或 `count`。