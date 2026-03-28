"""
政府采购 P-Tuning 训练数据生成（分类 / 关键词 / NL2SQL）。

NL2SQL 已改为英文列名（与 config.cfg.SQL_EN_TO_ZH_COLUMNS、prompt_util.nl2sql_prompt_prefix 一致）。

重新生成英文 NL2SQL 数据并训练：
  1) 在 Code 目录下：  python ptuning/generate_procurement_ptuning_data.py
  2) 分类 / 关键词 / NL2SQL 均使用 pre_seq_len=128（与 cfg 一致），分别在各 ptuning/*/ 目录按 train.sh 训练。
  3) 训练完成后将各 output/.../pytorch_model.bin 对齐 cfg 中 CLASSIFY_/KEYWORDS_/NL2SQL_CHECKPOINT_PATH（详见 ptuning/README_SHARED_PREFIX.md）。
"""
import json
import random
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from prompt_util import nl2sql_prompt_prefix

RNG = random.Random(42)

FIELDS = [
    "公告标题",
    "项目编号",
    "发布日期",
    "详情页链接",
    "招标人/采购单位",
    "招标人联系人",
    "招标人联系方式",
    "招标代理机构",
    "招标代理机构联系人",
    "招标代理机构联系方式",
    "中标人",
    "中标人联系人",
    "中标人联系方式",
    "中标金额(元)",
    "招标内容",
    "项目地点",
    "招标文件",
    "抓取时间",
]

BUYERS = [
    "上海市浦东新区惠南镇人民政府",
    "上海市闵行区教育局",
    "上海市黄浦区卫生健康委员会",
    "上海市徐汇区城市管理行政执法局",
]
AGENCIES = [
    "上海亚圣建设工程造价咨询有限公司",
    "上海联合工程监理造价咨询有限公司",
    "上海建实财务监理有限公司",
    "上海申厚建设咨询事务所有限公司",
]
WINNERS = [
    "上海城建市政工程（集团）有限公司",
    "上海电气智慧城市科技有限公司",
    "中电科数字技术股份有限公司",
    "上海东方明珠新媒体股份有限公司",
]
LOCATIONS = ["浦东新区", "黄浦区", "徐汇区", "闵行区", "宝山区", "杨浦区"]


def classify_prompt(q: str) -> str:
    return """
        请问“{}”是属于下面哪个类别的问题?
        A: 招标/采购基础信息查询。
        B: 中标结果明细查询。
        C: 条件过滤下的明细查询（仍返回具体记录）。
        D: 计算题（公式推导类）。
        E: 统计/排序/聚合检索题（SQL检索类）。
        F: 开放性问题。
        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.
        """.format(q)


def keywords_prompt(q: str) -> str:
    return """
        请帮我从以下句子中提取关键词。这些关键词是句子中最重要、最能概括句子主题的词汇。通过这些关键词,你可以更好地理解句子的内容。你只需要回答文本中的关键词,不要回答其他内容.
        用户输入：
        "{}"
        """.format(q)


def build_nl2sql_instruction(user_q: str) -> str:
    """与 chatglm_ptuning.nl2sql 拼接一致：前缀 + \"问题\" """
    q = (user_q or "").strip().replace('"', "＂")
    return f'{nl2sql_prompt_prefix()}"{q}"'


def classify_samples():
    rows = []
    q_templates = [
        ("A", "项目编号{}的公告标题和发布日期是什么？"),
        ("A", "{}这个项目的招标人/采购单位和招标代理机构是谁？"),
        ("B", "项目编号{}是谁中标？中标金额是多少？"),
        ("B", "{}这个项目的中标人联系方式是什么？"),
        ("C", "{}在{}发布的项目里，中标人分别是谁？"),
        ("C", "由{}代理且项目地点在{}的公告有哪些？"),
        ("D", "{}在{}年的中标金额同比增长率是多少？"),
        ("D", "{}在{}年中标金额占比是多少？"),
        ("E", "{}年{}代理了多少个项目？"),
        ("E", "{}年{}发布项目中中标金额前3的是哪些？"),
        ("F", "请简要分析{}近期政府采购项目特点。"),
        ("F", "什么是政府采购中的公开招标？"),
    ]
    years = ["2024", "2025", "2026"]
    for i in range(2400):
        label, tpl = q_templates[i % len(q_templates)]
        buyer = BUYERS[i % len(BUYERS)]
        agency = AGENCIES[i % len(AGENCIES)]
        winner = WINNERS[i % len(WINNERS)]
        loc = LOCATIONS[i % len(LOCATIONS)]
        proj = "3101{:02d}51251226162488-{:03d}".format(i % 99, 100 + (i % 900))
        year = years[i % len(years)]
        q = tpl.format(proj, buyer, agency, loc, winner, year)
        rows.append(
            {"id": i, "question_prompt": classify_prompt(q), "question": q, "query": label}
        )
    return rows


def keywords_from_question(q: str, label: str) -> str:
    base = ["TYPE_{}".format(label)]
    for field in FIELDS:
        if field in q:
            base.append(field)
    if "采购人" in q or "招标人" in q:
        base.append("招标人/采购单位")
    if "中标金额" in q:
        base.append("中标金额(元)")
    if "代理" in q:
        base.append("招标代理机构")
    if "项目地点" in q or any(loc in q for loc in LOCATIONS):
        base.append("项目地点")
    uniq = []
    seen = set()
    for x in base:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return ",".join(uniq[:8])


def keywords_samples(classify_rows):
    rows = []
    for i, item in enumerate(classify_rows):
        rows.append(
            {
                "id": i,
                "question_prompt": keywords_prompt(item["question"]),
                "question": item["question"],
                "query": keywords_from_question(item["question"], item["query"]),
            }
        )
    return rows


def nl2sql_samples():
    rows = []
    for i in range(3000):
        year = 2024 + (i % 3)
        month = 1 + (i % 12)
        buyer = BUYERS[i % len(BUYERS)]
        agency = AGENCIES[i % len(AGENCIES)]
        winner = WINNERS[i % len(WINNERS)]
        loc = LOCATIONS[i % len(LOCATIONS)]
        proj = "3101{:02d}51251226162488-{:03d}".format(i % 99, 100 + (i % 900))
        mode = i % 6
        if mode == 0:
            q = "{}年{}月{}发布项目中中标金额总和是多少？".format(year, month, buyer)
            sql = "select sum(`bid_amount`) as total_bid_amount from shggzy_bid_result where DATE_FORMAT(`publish_date`, '%Y-%m')='{}-{:02d}' and `bidder`='{}'".format(year, month, buyer)
        elif mode == 1:
            q = "项目编号{}的中标人和详情链接是什么？".format(proj)
            sql = "select `project_no`,`winner`,`detail_url` from shggzy_bid_result where `project_no`='{}' limit 1".format(proj)
        elif mode == 2:
            q = "{}年{}代理了多少个项目？".format(year, agency)
            sql = "select count(distinct `project_no`) as project_cnt from shggzy_bid_result where DATE_FORMAT(`publish_date`, '%Y')='{}' and `agency`='{}'".format(year, agency)
        elif mode == 3:
            q = "{}年{}中标项目数量是多少？".format(year, winner)
            sql = "select count(1) as win_cnt from shggzy_bid_result where DATE_FORMAT(`publish_date`, '%Y')='{}' and `winner`='{}'".format(year, winner)
        elif mode == 4:
            q = "{}项目地点在{}的公告标题有哪些？".format(year, loc)
            sql = "select `title`,`project_location` from shggzy_bid_result where DATE_FORMAT(`publish_date`, '%Y')='{}' and `project_location` like '%{}%' limit 20".format(year, loc)
        else:
            q = "{}年中标金额前5的项目编号和中标金额分别是多少？".format(year)
            sql = "select `project_no`,`bid_amount` from shggzy_bid_result where DATE_FORMAT(`publish_date`, '%Y')='{}' order by `bid_amount` desc limit 5".format(year)
        rows.append({"question": build_nl2sql_instruction(q), "answer": "```sql\n{}\n```".format(sql)})
    return rows


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    root = Path(__file__).resolve().parent
    cls = classify_samples()
    kw = keywords_samples(cls)
    nl = nl2sql_samples()

    write_json(root / "CLASSIFY_PTUNING" / "Fin_train" / "train.json", cls[:2200])
    write_json(root / "CLASSIFY_PTUNING" / "Fin_train" / "dev.json", cls[2200:])
    # keywords 与 classify 同源条数（当前 2400）；不能用 2600 切分，否则 dev 会变成 []
    write_json(root / "KEYWORDS_PTUNING" / "Fin_train" / "train.json", kw[:2200])
    write_json(root / "KEYWORDS_PTUNING" / "Fin_train" / "dev.json", kw[2200:])
    write_jsonl(root / "NL2SQL_PTUNING" / "train_data" / "nl2sql_train_data.json", nl[:2700])
    write_jsonl(root / "NL2SQL_PTUNING" / "train_data" / "nl2sql_dev_data.json", nl[2700:])
    print("generated classify={}, keywords={}, nl2sql={}".format(len(cls), len(kw), len(nl)))


if __name__ == "__main__":
    main()
