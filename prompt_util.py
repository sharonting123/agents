# -*- coding: utf-8 -*-
"""
NL2SQL 纠错 / 字段对齐用的提示词模板（.format 占位符与 sql_correct_util、answers/sql_answer 一致）。
执行层 MySQL 列名为英文（见 cfg.SQL_EN_TO_ZH_COLUMNS）；NL2SQL 输出应使用英文列名。
"""


def build_sql_column_catalog() -> str:
    """英文列名 + 中文注释，供 NL2SQL / SQL 纠错提示词使用。"""
    from config import cfg

    cmap = getattr(cfg, "SQL_EN_TO_ZH_COLUMNS", None) or {}
    return "\n".join(f"  `{en}`  -- {zh}" for en, zh in sorted(cmap.items(), key=lambda x: x[0]))


def build_sql_column_csv_english() -> str:
    """仅英文列名逗号分隔，供简短「已知字段」一行。"""
    from config import cfg

    cmap = getattr(cfg, "SQL_EN_TO_ZH_COLUMNS", None) or {}
    return ",".join(f"`{en}`" for en in sorted(cmap.keys()))


def nl2sql_prompt_prefix() -> str:
    """
    NL2SQL P-Tuning 与线上推理共用的系统提示前缀（不含用户问题）。
    与 chatglm_ptuning.ChatGLM_Ptuning.nl2sql 拼接方式：f'{prefix}\"{question}\"'
    """
    field_catalog = build_sql_column_catalog()
    field_one_line = build_sql_column_csv_english()
    return f"""你是一名 MySQL 开发人员。根据表结构与用户问题编写 SQL。**列名必须使用下列英文标识符**（反引号包裹），不要用中文列名；表名：`shggzy_bid_result`。

已知字段一览（English `column` -- 中文含义）：
{field_catalog}

字段名速查（仅英文）：{field_one_line}

注意：把问题中的中文数量词转为阿拉伯数字（如一亿→100000000，一千万→10000000，两千万→20000000 等）。
不得使用上表以外的列名。

示例（SQL 中列名均为英文）：

用户输入：2026年3月上海市浦东新区惠南镇人民政府在公开招标上一共中标了多少金额？

sql如下：
```sql 
select sum(`bid_amount`) as total_bid_amount
from shggzy_bid_result
where DATE_FORMAT(`publish_date`, '%Y-%m') = '2026-03'
  and `bidder` = '上海市浦东新区惠南镇人民政府'
limit 1
```

用户输入：项目编号310115131251226162488-15301828谁中标了？公告链接在哪？

sql如下：
```sql 
select `project_no`,`winner`,`detail_url`
from shggzy_bid_result
where `project_no` = '310115131251226162488-15301828'
limit 1
```

用户输入：2026年上海亚圣建设工程造价咨询有限公司一共代理了多少个招标？

sql如下：
```sql
select count(distinct `project_no`) as project_cnt
from shggzy_bid_result
where DATE_FORMAT(`publish_date`, '%Y') = '2026'
  and `agency` = '上海亚圣建设工程造价咨询有限公司'
```

请根据以下用户输入，只输出 ```sql 代码块（列名为英文）。
用户输入："""


# get_most_like_word：.format(候选列表, 查询词)
# 解析见 sql_correct_util：需在输出中包含「查询词语：」+ 选中的词
prompt_most_like_word = """从下列候选字段名中，选出与「查询词」含义最接近的一个（必须原样来自候选列表）。

候选字段名列表：{}
查询词：{}

只输出一行：查询词语：<你选中的候选词>
若无法匹配则输出：查询词语：无
"""

# SQL 执行失败时让底座模型改 SQL：.format(合法字段说明, 当前sql, 报错信息)
# 第一个占位符请传入 build_sql_column_catalog()（英文列名 + 中文注释）
prompt_sql_correct = """你是 SQL 纠错助手。真实表名：`shggzy_bid_result`（提示词里也可能写作 company_table，执行时会映射）。请**只使用下列英文列名**（反引号包裹），不要自造列名：

{}

当前 SQL：
{}

数据库执行报错：
{}

请根据报错修正 SQL，只输出一个 Markdown 代码块，格式如下：
```sql
（修正后的完整一条 SQL，列名须为英文标识符）
```
不要输出其它解释。"""
