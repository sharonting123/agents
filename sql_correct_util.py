# [18] 阅读顺序编号文件，对应原文件: sql_correct_util.py
import company_table
import re
import prompt_util
from loguru import logger
from config import cfg


def normalize_sql_for_mysql(sql: str) -> str:
    """将 company_table 映射为真实表名；若 SQL 仍为中文列名则替换为 MySQL 英文列（cfg.SQL_EN_TO_ZH_COLUMNS）。NL2SQL 现以英文列为主，本函数对纯英文 SQL 多为恒等。"""
    if not getattr(cfg, "USE_MYSQL_FOR_SQL", False):
        return sql
    s = sql.strip()
    table = getattr(cfg, "MYSQL_SQL_TABLE", "shggzy_bid_result")
    s = re.sub(r"(?i)\bcompany_table\b", "`{}`".format(table), s)
    # 英文列名 <- 中文注释：模型 SQL 多为 `中文列`；若只对中文子串 replace 会得到 `` `en` `` 双反引号，MySQL 1064
    cmap = getattr(cfg, "SQL_EN_TO_ZH_COLUMNS", None) or {}
    for en, zh in sorted(cmap.items(), key=lambda x: -len(x[1])):
        target = "`{}`".format(en)
        quoted_zh = "`{}`".format(zh)
        if quoted_zh in s:
            s = s.replace(quoted_zh, target)
        elif zh in s:
            s = s.replace(zh, target)
    # 年报式「年份 = '2021'」→ 按发布日期年份过滤
    s = re.sub(r"年份\s*=\s*['\"]?(\d{4})['\"]?", r"YEAR(`publish_date`) = \1", s)
    return s


def exc_sql(ori_question, sql, sql_cursor):
    answer = None
    exec_log = ""
    sql = normalize_sql_for_mysql(sql)
    try:
        sql_cursor.execute(sql)
        result = sql_cursor.fetchall()
        rows = []
        for row in result[:50]:
            vals = []
            for val in row:
                if val is None:
                    vals.append("NULL")
                    continue
                try:
                    num = float(val)
                    # 聚合/计数结果不要用「元/个/家」混合格式，避免误读与大数问题
                    if num == int(num) and abs(num) < 1e15:
                        vals.append(str(int(num)))
                    else:
                        vals.append("{:.4f}".format(num).rstrip("0").rstrip("."))
                except (TypeError, ValueError):
                    vals.append(str(val))
            rows.append(",".join(vals))
        # 仅返回查询结果，不再把原问题拼在前面（否则前端会显示「问题全文 + 数字」）
        if rows:
            answer = ";".join(rows)
        else:
            answer = "查询无结果"
    except Exception as e:
        logger.error('执行SQL[{}]错误! {}'.format(sql.replace('<>', ''), e))
        exec_log = str(e)
    return answer, exec_log


def _desc_column_name(desc_item) -> str:
    """pymysql / sqlite3 cursor.description 列名。"""
    if not desc_item:
        return ""
    n = desc_item[0]
    if isinstance(n, (bytes, bytearray)):
        return n.decode("utf-8", errors="replace")
    return str(n)


def exc_sql_rows(ori_question, sql, sql_cursor):
    """
    执行 SQL，返回结构化行（最多 50 行）供图表与数据分析。
    返回 (rows, columns, exec_log)；失败时 rows 为 None。
    """
    _ = ori_question
    sql = normalize_sql_for_mysql(sql)
    try:
        sql_cursor.execute(sql)
        result = sql_cursor.fetchall()
        desc = sql_cursor.description
        if not desc:
            return [], [], ""
        cols = [_desc_column_name(d) for d in desc]
        rows = []
        for row in result[:50]:
            rows.append({cols[i]: row[i] for i in range(len(cols))})
        return rows, cols, ""
    except Exception as e:
        logger.error('执行SQL[{}]错误! {}'.format(sql.replace('<>', ''), e))
        return None, None, str(e)


def get_field_number(sql):
    sql_words = sql.split(' ')
    fields = []
    numbers = []
    pre_word = ''
    for word in sql_words:
        if word == '' or word in ['(', ')']:
            continue
        if word.startswith('('):
            word = word[1:]
        # 只检查条件字段
        if pre_word in ['and', 'or', 'by', 'where'] and re.match(r'^[\u4E00-\u9FA5]+$', word):
            fields.append(word)
        elif pre_word in ['<', '>'] and re.match(r'^[0-9]+$', word) and len(word) > 2:
            numbers.append(word)
        pre_word = word
    return fields, numbers

def get_number_from_question(question):
    unit_dic = {'十万': 100000, '百万': 1000000, '千万': 10000000, '十亿': 1000000000, '百亿': 10000000000,
                '千亿': 100000000000, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
    num_dic = {"一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

    numbers = re.findall('([一二三四五六七八九十两1234567890]+个?(十万|百万|千万|十亿|百亿|千亿|百|千|万|亿|))', question)
    number_list = []
    for number in numbers:
        # print(number)
        digit_num = number[0].replace('个', '')
        if len(number[1]) > 0:
            digit_num = digit_num.replace(number[1], '')
        if len(digit_num) > 0 and digit_num[-1] in ['十', '百', '千', '万']:
            print(digit_num)
            unit = digit_num[-1] + number[1]
            digit_num = digit_num[:-1]
        else:
            unit = number[1]
        # 太小的纯数字和年份不作检查
        if unit == '' and (len(digit_num) < 3 or (len(digit_num) == 4 and digit_num[:2] == '20')):
            continue
        # 纯数字，不带单位
        elif unit == '' and re.match('^[0-9]+$', digit_num):
            number_list.append(digit_num)
        # 十亿、百亿类直接是单位
        elif digit_num == '' and len(unit) == 2 and unit in unit_dic.keys():
            number_list.append(str(unit_dic.get(unit)))
        # 带单位
        elif unit in unit_dic.keys():
            digit_num = digit_num.replace(unit, '')
            if digit_num in num_dic.keys():
                digit_num = num_dic.get(digit_num)
                number_list.append(str(digit_num*unit_dic.get(unit)))
            elif re.match('^[0-9]+$', digit_num):
                number_list.append(str(int(digit_num) * unit_dic.get(unit)))
    return number_list


def get_most_like_word(word, word_lsit, model):
    mst_like_word = ''
    try:
        answer = model(prompt_util.prompt_most_like_word.format(word_lsit, word))
        logger.info('同义词查询：{}'.format(answer.replace('<>', '')))
    except Exception as e:
        logger.warning('模型查询同义词字段失败：{}'.format(str(e).replace('<>', '')))
    if '查询词语：' in answer:
        most_like_word = answer[answer.find('查询词语：')+5:]
        if most_like_word in word_lsit:
            mst_like_word = most_like_word
    return mst_like_word


def correct_sql_field(sql, question, model):
    new_sql = sql
    key_words = list(company_table.load_company_table().columns)

    fields, sql_numbers = get_field_number(sql)
    for field in fields:
        if field not in key_words:
            most_like_word = get_most_like_word(field, key_words,model)
            if len(most_like_word) > 0:
                logger.info('文本字段纠正前sql：{}'.format(new_sql))
                new_sql = new_sql.replace(field, most_like_word)
                logger.info('文本字段纠正后sql：{}'.format(new_sql))
    return new_sql


def correct_sql_number(sql, question):
    new_sql = sql
    fields, sql_numbers = get_field_number(sql)
    q_numbers = get_number_from_question(question)
    for sql_number in sql_numbers:
        if len(sql_number) > 2 and sql_number not in q_numbers and len(q_numbers) == 1:
            logger.info('文本数字纠正前sql：{}'.format(new_sql))
            new_sql = new_sql.replace(sql_number, q_numbers[0])
            logger.info('文本数字纠正后sql：{}'.format(new_sql))
    return new_sql


if __name__ == '__main__':
    sql = "select count(1), afdas from company_table where 年份 = '2020' and 公司注册地址 is not null and 归属于母公司所有者权益合计 < 50000000"
    question = '归属于母公司所有者权益合计大于500万'
    new_sql = correct_sql_number(sql,question)
    print(new_sql)
