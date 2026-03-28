"""
上海市公共资源交易中心 - 政府采购公告爬虫
目标: 抓取 采购公告(channelId=37) + 中标结果公告(channelId=38)
目标页面: https://www.shggzy.com/search/queryContents.jhtml
运行方式: python shggzy_crawler.py
定时执行: 每天自动运行一次（内置调度，或用 cron/任务计划调用）

依赖安装:
    pip install requests beautifulsoup4 pymysql schedule loguru fake-useragent PyJWT
"""

import re
import time
import json
import random
import base64
import hashlib
import html as html_lib
from urllib.parse import urlsplit, parse_qsl, urlencode, urlunsplit
import schedule
import pymysql
import requests
import jwt                          # PyJWT
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup
from loguru import logger
from fake_useragent import UserAgent

# ─────────────────────────────────────────────
#  ① 配置区（按需修改）
# ─────────────────────────────────────────────
MYSQL_CONFIG = {
    "host":     "39.105.216.24",
    "port":     3306,
    "user":     "ztt",
    "password": "Ti123456!",   # ← 改成你的 MySQL 密码
    "database": "bidding",         # ← 数据库名（会自动创建）
    "charset":  "utf8mb4",
}

BASE_URL    = "https://www.shggzy.com"
SEARCH_URL  = f"{BASE_URL}/search/queryContents.jhtml"

# 要抓取的频道列表（支持多个）
# channelId=37: 采购公告
# channelId=38: 中标结果公告（政府采购）
CHANNEL_IDS = [37, 38]    # 同时抓取 采购公告 + 中标结果公告
CHANNEL_NAMES = {
    37: "采购公告",
    38: "中标结果公告",
}

IN_DATES    = 4000      # 全量历史抓取窗口（按站点支持范围）
PAGE_SIZE   = 20          # 每页条数（平台固定）
MAX_PAGES   = 1000        # 最大翻页数（全量建议 >= 1000）
REQUEST_INTERVAL = (2, 4) # 随机请求间隔（秒）
REQUEST_TIMEOUT = 25       # 单次请求超时（秒）
REQUEST_RETRIES = 3        # 列表/详情请求重试次数
INCREMENTAL_STOP_ON_DUP_PAGE = False  # True:增量模式遇全重复页即停; False:全量继续翻页
FIRST_PAGE_VALIDATE_GATE = True       # 先抓第一页并校验，再继续翻页
TARGET_DATE_OFFSET_DAYS = 1           # 目标日期偏移（1=昨天）
SCHEDULE_TIME = "07:00"               # 每天定时运行时间
RUN_ON_START = False                  # 启动时是否立即跑一次

# ─────────────────────────────────────────────
#  ② 工具函数
# ─────────────────────────────────────────────
ua = UserAgent()
DESKTOP_FALLBACK_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

def _desktop_ua() -> str:
    """避免拿到移动端 UA（会导致详情链接失真或返回空内容）"""
    try:
        candidate = ua.random
    except Exception:
        return DESKTOP_FALLBACK_UA
    lower = candidate.lower()
    if any(k in lower for k in ("mobile", "iphone", "android", "ipad")):
        return DESKTOP_FALLBACK_UA
    return candidate

def _headers() -> dict:
    """每次请求随机换 UA，模拟正常浏览器行为"""
    return {
        "User-Agent":      _desktop_ua(),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         BASE_URL,
        "Connection":      "keep-alive",
    }

def _make_cext(page_no: int) -> str:
    """
    cExt 是一个 HS256 JWT，payload 包含 path / pageNo / exp。
    平台用固定密钥签名；通过抓包发现密钥为空字符串（即 alg=none 变体）。
    若平台升级了密钥，直接把 cExt 留空——大多数情况下 pageNo 参数单独生效。
    """
    payload = {
        "path":   "/jyxxzc",
        "pageNo": page_no,
        "exp":    int(time.time() * 1000) + 86400_000,  # 24h 后过期
    }
    try:
        token = jwt.encode(payload, "", algorithm="HS256")
        return token if isinstance(token, str) else token.decode()
    except Exception:
        # PyJWT 版本差异兜底
        header  = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode()
        body    = base64.urlsafe_b64encode(
                    json.dumps(payload, separators=(",", ":")).encode()
                  ).rstrip(b"=").decode()
        return f"{header}.{body}."

def _sleep():
    """礼貌等待"""
    time.sleep(random.uniform(*REQUEST_INTERVAL))


def _target_date() -> date:
    """返回本轮要抓取的目标日期（默认昨天）"""
    return date.today() - timedelta(days=TARGET_DATE_OFFSET_DAYS)


def _clean_contact_name(name: str | None) -> str | None:
    """
    联系人字段清洗：仅保留姓名（首个空格前内容）。
    例如：'张三 13800001111' -> '张三'
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    # 统一全角空格并按任意空白切分，只保留第一个片段
    s = s.replace("\u3000", " ")
    parts = re.split(r"\s+", s)
    first = parts[0].strip() if parts else s
    return first or None


def _split_bidder_fields(raw_bidder: str | None) -> tuple[str | None, str | None, str | None]:
    """
    从 bidder 原始文本中拆分：
      - 公司名称（bidder）
      - 地址（bidder_contact 字段复用为地址）
      - 联系方式（bidder_phone）
    """
    if not raw_bidder:
        return None, None, None

    text = re.sub(r"\s+", " ", str(raw_bidder).replace("\u3000", " ").strip())
    if not text:
        return None, None, None

    address = None
    phone = None
    working = text
    # 标签归一化，适配“地 址 / 联系 方 式”这类空格写法
    working = re.sub(r"地\s*址", "地址", working)
    working = re.sub(r"联\s*系\s*方\s*式", "联系方式", working)
    working = re.sub(r"联\s*系\s*电\s*话", "联系电话", working)
    working = re.sub(r"项\s*目\s*联\s*系\s*人", "项目联系人", working)

    m_phone = re.search(
        r"(?:联系方式|联系电话|电话|联系方式电话)[：:\s]*([0-9\-（）()]{7,25})",
        working,
        flags=re.IGNORECASE,
    )
    if m_phone:
        phone = re.sub(r"[^\d\-]", "", m_phone.group(1))
        working = working.replace(m_phone.group(0), " ")
    else:
        # 无明确标签时的兜底：仅在文本中包含“电话/联系方式”字样时启用
        if re.search(r"(电话|联系方式)", working):
            m_plain_phone = re.search(r"([0-9\-（）()]{7,25})", working)
            if m_plain_phone:
                phone = re.sub(r"[^\d\-]", "", m_plain_phone.group(1))
                working = working.replace(m_plain_phone.group(1), " ")

    m_addr = re.search(
        r"(?:地址|联系地址|通讯地址)[：:\s]*([^；;，,。]{2,180}?)(?=(?:联系方式|联系电话|电话|联系人|；|;|，|,|。|$))",
        working,
        flags=re.IGNORECASE,
    )
    if m_addr:
        address = m_addr.group(1).strip(" ：:;；,，。")
        working = working.replace(m_addr.group(0), " ")

    # 去掉前缀标签，保留公司名称主体
    working = re.sub(r"^(?:采购人(?:名称)?|招标人(?:名称)?|单位名称)[：:\s]*", "", working).strip()
    # 遇到地址/联系方式标签时截断后半段
    cut_keywords = ["地址", "联系地址", "通讯地址", "联系方式", "联系电话", "电话", "联系人"]
    cut_pos = len(working)
    for k in cut_keywords:
        i = working.find(k)
        if i >= 0:
            cut_pos = min(cut_pos, i)
    bidder_name = working[:cut_pos].strip(" ：:;；,，。")
    # 再按分隔符截一次，避免拖尾信息
    bidder_name = re.split(r"[；;，,。]", bidder_name)[0].strip() if bidder_name else None
    bidder_name = re.sub(r"\s*地址[：:].*$", "", bidder_name or "").strip()
    bidder_name = re.sub(r"\s*联系方式[：:].*$", "", bidder_name or "").strip()

    return bidder_name or None, address or None, phone or None


def _fit_varchar(value: str | None, max_len: int) -> str | None:
    """防止写库时字段超长。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s[:max_len]


def _normalize_project_no(value: str | None) -> str | None:
    """规范化项目编号，支持英文+数字+短横线等组合。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"^(?:项目编号|采购编号|招标编号|编号)[：:\s]*", "", s, flags=re.IGNORECASE)
    s = s.strip("：:;；,，。")
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_./]{3,120}", s):
        return None
    return s[:100]


def _extract_project_no_from_text(text: str | None) -> str | None:
    """从文本中提取最可能的项目编号。"""
    if not text:
        return None
    t = str(text).replace("\u3000", " ")

    m = re.search(
        r"(?:项目编号|采购编号|招标编号|项目编码|编号)[：:\s]*([A-Za-z0-9][A-Za-z0-9\-_./]{3,120})",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        v = _normalize_project_no(m.group(1))
        if v:
            return v

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_./]{5,120}", t)
    best = None
    best_score = -1
    for tok in tokens:
        v = _normalize_project_no(tok)
        if not v:
            continue
        if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", v):
            continue
        score = 0
        if "-" in v:
            score += 3
        if re.search(r"[A-Za-z]", v) and re.search(r"\d", v):
            score += 3
        if len(v) >= 18:
            score += 2
        if re.search(r"\d{6,}", v):
            score += 1
        if score > best_score:
            best_score = score
            best = v
    return best


def _looks_like_address(text: str | None) -> bool:
    if not text:
        return False
    s = str(text)
    keys = ["地址", "省", "市", "区", "县", "路", "街", "巷", "号", "弄", "室", "楼", "层", "大道"]
    return any(k in s for k in keys)


def _normalize_agency_contact(value: str | None) -> str | None:
    """
    agency_contact 兼容两种来源：
      - 地址（采购公告）
      - 联系人姓名（中标结果）
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if _looks_like_address(s):
        return s
    return _clean_contact_name(s)


def _split_agency_fields(raw_agency: str | None) -> tuple[str | None, str | None, str | None]:
    """
    从 agency 原始文本中拆分：
      - 机构名称（agency）
      - 地址（agency_contact 字段复用为地址）
      - 联系方式（agency_phone）
    """
    if not raw_agency:
        return None, None, None

    text = re.sub(r"\s+", " ", str(raw_agency).replace("\u3000", " ").strip())
    if not text:
        return None, None, None

    text = re.sub(r"地\s*址", "地址", text)
    text = re.sub(r"联\s*系\s*方\s*式", "联系方式", text)
    text = re.sub(r"联\s*系\s*电\s*话", "联系电话", text)

    address = None
    phone = None
    working = text

    m_phone = re.search(r"(?:联系方式|联系电话|电话)[：:\s]*([0-9\-（）()]{7,25})", working, flags=re.IGNORECASE)
    if m_phone:
        phone = re.sub(r"[^\d\-]", "", m_phone.group(1))
        working = working.replace(m_phone.group(0), " ")

    m_addr = re.search(
        r"(?:地址|联系地址|通讯地址)[：:\s]*([^；;，,。]{2,180}?)(?=(?:联系方式|联系电话|电话|联系人|；|;|，|,|。|$))",
        working,
        flags=re.IGNORECASE,
    )
    if m_addr:
        address = m_addr.group(1).strip(" ：:;；,，。")
        working = working.replace(m_addr.group(0), " ")

    working = re.sub(r"^(?:采购代理机构(?:名称)?|招标代理机构(?:名称)?|代理机构(?:名称)?)[：:\s]*", "", working).strip()
    cut_keywords = ["地址", "联系地址", "通讯地址", "联系方式", "联系电话", "电话", "联系人"]
    cut_pos = len(working)
    for k in cut_keywords:
        i = working.find(k)
        if i >= 0:
            cut_pos = min(cut_pos, i)
    agency_name = working[:cut_pos].strip(" ：:;；,，。")
    agency_name = re.split(r"[；;，,。]", agency_name)[0].strip() if agency_name else None
    agency_name = re.sub(r"\s*地址[：:].*$", "", agency_name or "").strip()
    agency_name = re.sub(r"\s*联系方式[：:].*$", "", agency_name or "").strip()

    return agency_name or None, address or None, phone or None

# ─────────────────────────────────────────────
#  ③ 数据库初始化
# ─────────────────────────────────────────────
def init_db():
    """自动建库 + 建表（幂等）"""
    # 先不指定 database 连接，以便创建库
    conn = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        charset=MYSQL_CONFIG["charset"],
    )
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}` "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
    conn.commit()
    conn.close()

    # 正式连接目标库，创建两个独立表
    conn = _get_conn()
    with conn.cursor() as cur:
        # 采购公告表 (channelId=35)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `shggzy_purchase` (
                `id`           BIGINT       NOT NULL AUTO_INCREMENT,
                `title`        VARCHAR(500) NOT NULL COMMENT '公告标题',
                `project_no`   VARCHAR(100) DEFAULT NULL COMMENT '项目编号',
                `publish_date` DATE         DEFAULT NULL COMMENT '发布日期',
                `detail_url`   VARCHAR(1000) NOT NULL COMMENT '详情页链接',
                `bidder`       VARCHAR(200) DEFAULT NULL COMMENT '采购人名称',
                `bidder_contact` VARCHAR(200) DEFAULT NULL COMMENT '采购人地址',
                `bidder_phone` VARCHAR(50)  DEFAULT NULL COMMENT '采购人联系方式',
                `agency`       VARCHAR(200) DEFAULT NULL COMMENT '采购代理机构名称',
                `agency_contact` VARCHAR(200) DEFAULT NULL COMMENT '采购代理机构地址',
                `agency_phone` VARCHAR(50)  DEFAULT NULL COMMENT '采购代理机构联系方式',
                `project_contact` VARCHAR(100) DEFAULT NULL COMMENT '项目联系人',
                `project_phone` VARCHAR(50) DEFAULT NULL COMMENT '项目联系电话',
                `crawl_time`   DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '抓取时间',
                `url_hash`     CHAR(32)     NOT NULL COMMENT 'URL MD5去重键',
                PRIMARY KEY (`id`),
                UNIQUE KEY `uk_url_hash` (`url_hash`),
                KEY `idx_publish_date` (`publish_date`),
                KEY `idx_crawl_time`   (`crawl_time`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='上海公共资源交易中心-采购公告';
        """)
        
        # 中标结果公告表 (channelId=36) - 详细字段
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `shggzy_bid_result` (
                `id`                      BIGINT        NOT NULL AUTO_INCREMENT,
                `title`                   VARCHAR(500)  NOT NULL COMMENT '公告标题',
                `project_no`               VARCHAR(100)  DEFAULT NULL COMMENT '项目编号',
                `publish_date`             DATE          DEFAULT NULL COMMENT '发布日期',
                `detail_url`              VARCHAR(1000) NOT NULL COMMENT '详情页链接',
                `bidder`                  VARCHAR(200)  DEFAULT NULL COMMENT '招标人/采购单位',
                `bidder_contact`          VARCHAR(200)  DEFAULT NULL COMMENT '招标人地址',
                `bidder_phone`            VARCHAR(50)   DEFAULT NULL COMMENT '招标人联系方式',
                `agency`                  VARCHAR(200)  DEFAULT NULL COMMENT '招标代理机构',
                `agency_contact`          VARCHAR(200)  DEFAULT NULL COMMENT '招标代理机构联系人/地址',
                `agency_phone`            VARCHAR(50)   DEFAULT NULL COMMENT '招标代理机构联系方式',
                `winner`                  VARCHAR(200)  DEFAULT NULL COMMENT '中标人',
                `winner_contact`          VARCHAR(50)   DEFAULT NULL COMMENT '中标人联系人',
                `winner_phone`            VARCHAR(50)   DEFAULT NULL COMMENT '中标人联系方式',
                `bid_amount`              DECIMAL(20,2) DEFAULT NULL COMMENT '中标金额(元)',
                `agency_service_fee_amount` DECIMAL(20,2) DEFAULT NULL COMMENT '代理服务收费金额(元)',
                `bid_content`             TEXT           DEFAULT NULL COMMENT '招标内容',
                `project_location`         VARCHAR(200)  DEFAULT NULL COMMENT '项目地点',
                `tender_file`             VARCHAR(500)  DEFAULT NULL COMMENT '招标文件',
                `crawl_time`              DATETIME      DEFAULT CURRENT_TIMESTAMP COMMENT '抓取时间',
                `url_hash`                CHAR(32)      NOT NULL COMMENT 'URL MD5去重键',
                PRIMARY KEY (`id`),
                UNIQUE KEY `uk_url_hash` (`url_hash`),
                KEY `idx_publish_date`    (`publish_date`),
                KEY `idx_bidder`          (`bidder`),
                KEY `idx_winner`          (`winner`),
                KEY `idx_crawl_time`      (`crawl_time`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='上海公共资源交易中心-中标结果公告';
        """)
        # 兼容已有旧表：补齐新增字段（代理服务收费金额）
        try:
            cur.execute("""
                ALTER TABLE `shggzy_bid_result`
                ADD COLUMN `agency_service_fee_amount` DECIMAL(20,2) DEFAULT NULL COMMENT '代理服务收费金额(元)'
            """)
        except Exception:
            # 字段已存在时忽略
            pass
        # 兼容旧表：bidder_contact 由“联系人”改为“地址”，并放宽长度
        try:
            cur.execute("""
                ALTER TABLE `shggzy_bid_result`
                MODIFY COLUMN `bidder_contact` VARCHAR(200) DEFAULT NULL COMMENT '招标人地址'
            """)
        except Exception:
            pass
        # 兼容旧表：agency_contact 放宽长度，兼容地址场景
        try:
            cur.execute("""
                ALTER TABLE `shggzy_bid_result`
                MODIFY COLUMN `agency_contact` VARCHAR(200) DEFAULT NULL COMMENT '招标代理机构联系人/地址'
            """)
        except Exception:
            pass
        # 兼容旧采购公告表：补齐详情字段
        for ddl in [
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `bidder` VARCHAR(200) DEFAULT NULL COMMENT '采购人名称'",
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `bidder_contact` VARCHAR(200) DEFAULT NULL COMMENT '采购人地址'",
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `bidder_phone` VARCHAR(50) DEFAULT NULL COMMENT '采购人联系方式'",
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `agency` VARCHAR(200) DEFAULT NULL COMMENT '采购代理机构名称'",
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `agency_contact` VARCHAR(200) DEFAULT NULL COMMENT '采购代理机构地址'",
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `agency_phone` VARCHAR(50) DEFAULT NULL COMMENT '采购代理机构联系方式'",
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `project_contact` VARCHAR(100) DEFAULT NULL COMMENT '项目联系人'",
            "ALTER TABLE `shggzy_purchase` ADD COLUMN `project_phone` VARCHAR(50) DEFAULT NULL COMMENT '项目联系电话'",
        ]:
            try:
                cur.execute(ddl)
            except Exception:
                pass
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")

def _get_conn():
    return pymysql.connect(**MYSQL_CONFIG)

def _stable_url_for_hash(url: str) -> str:
    """
    生成稳定去重键：
    - 去掉动态参数（cExt/isIndex/ext2 等）
    - 保留业务参数
    """
    if not url:
        return url
    parts = urlsplit(url)
    drop_keys = {"cext", "isindex", "ext2", "_", "timestamp", "t"}
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in drop_keys]
    q.sort(key=lambda x: (x[0], x[1]))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(q, doseq=True), ""))

def _url_hash(url: str) -> str:
    stable = _stable_url_for_hash(url)
    return hashlib.md5(stable.encode()).hexdigest()

def _normalize_detail_url(url: str) -> str:
    """
    详情链接规范化：
    - 移动页常给 /jyxxzcgs/{id}?ext2=
    - 实测带 cExt/isIndex 的桌面链接稳定返回完整详情
    """
    if not url:
        return url
    parts = urlsplit(url)
    if "/jyxxzcgs/" not in parts.path:
        return url
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    if "cExt" not in q:
        q.pop("ext2", None)
        q["cExt"] = _make_cext(1)
    q.setdefault("isIndex", "")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q, doseq=True), parts.fragment))

def save_records(records: list[dict], channel_id: int, detail_data: list[dict] = None) -> tuple[int, int]:
    """
    批量插入，跳过已存在的记录（按 url_hash 去重）。
    根据 channel_id 写入不同的表：
      - 37 -> shggzy_purchase (采购公告)
      - 38 -> shggzy_bid_result (中标结果公告) - 带详细字段
    detail_data: 详情页解析后的数据列表，与 records 一一对应
    返回 (新增数量, 跳过数量)
    """
    if not records:
        return 0, 0
    
    # 根据 channel_id 选择目标表
    if channel_id == 37:
        table_name = "shggzy_purchase"
        conn = _get_conn()
        inserted = skipped = 0
        sql = f"""
            INSERT INTO `{table_name}`
                (title, project_no, publish_date, detail_url, url_hash,
                 bidder, bidder_contact, bidder_phone,
                 agency, agency_contact, agency_phone,
                 project_contact, project_phone)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                project_no = VALUES(project_no),
                publish_date = VALUES(publish_date),
                detail_url = VALUES(detail_url),
                bidder = COALESCE(NULLIF(VALUES(bidder), ''), bidder),
                bidder_contact = COALESCE(NULLIF(VALUES(bidder_contact), ''), bidder_contact),
                bidder_phone = COALESCE(NULLIF(VALUES(bidder_phone), ''), bidder_phone),
                agency = COALESCE(NULLIF(VALUES(agency), ''), agency),
                agency_contact = COALESCE(NULLIF(VALUES(agency_contact), ''), agency_contact),
                agency_phone = COALESCE(NULLIF(VALUES(agency_phone), ''), agency_phone),
                project_contact = COALESCE(NULLIF(VALUES(project_contact), ''), project_contact),
                project_phone = COALESCE(NULLIF(VALUES(project_phone), ''), project_phone)
        """
        try:
            with conn.cursor() as cur:
                for i, r in enumerate(records):
                    h = _url_hash(r["detail_url"])
                    detail = detail_data[i] if detail_data and i < len(detail_data) else {}
                    affected = cur.execute(sql, (
                        r["title"],
                        r.get("project_no") or r.get("purchase_no"),
                        r.get("publish_date"),
                        r["detail_url"],
                        h,
                        _fit_varchar(detail.get("bidder"), 200),
                        _fit_varchar(detail.get("bidder_contact"), 200),
                        _fit_varchar(detail.get("bidder_phone"), 50),
                        _fit_varchar(detail.get("agency"), 200),
                        _fit_varchar(detail.get("agency_contact"), 200),
                        _fit_varchar(detail.get("agency_phone"), 50),
                        _clean_contact_name(detail.get("project_contact")),
                        _fit_varchar(detail.get("project_phone"), 50),
                    ))
                    if affected:
                        inserted += 1
                    else:
                        skipped += 1
            conn.commit()
        finally:
            conn.close()
        return inserted, skipped
    
    elif channel_id == 38:
        # 中标结果公告 - 带详细字段
        table_name = "shggzy_bid_result"
        conn = _get_conn()
        inserted = skipped = 0
        sql = f"""
            INSERT INTO `{table_name}` (
                title, project_no, publish_date, detail_url, url_hash,
                bidder, bidder_contact, bidder_phone,
                agency, agency_contact, agency_phone,
                winner, winner_contact, winner_phone,
                bid_amount, agency_service_fee_amount, bid_content, project_location, tender_file
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                project_no = VALUES(project_no),
                publish_date = VALUES(publish_date),
                detail_url = VALUES(detail_url),
                bidder = COALESCE(NULLIF(VALUES(bidder), ''), bidder),
                bidder_contact = COALESCE(NULLIF(VALUES(bidder_contact), ''), bidder_contact),
                bidder_phone = COALESCE(NULLIF(VALUES(bidder_phone), ''), bidder_phone),
                agency = COALESCE(NULLIF(VALUES(agency), ''), agency),
                agency_contact = COALESCE(NULLIF(VALUES(agency_contact), ''), agency_contact),
                agency_phone = COALESCE(NULLIF(VALUES(agency_phone), ''), agency_phone),
                winner = COALESCE(NULLIF(VALUES(winner), ''), winner),
                winner_contact = COALESCE(NULLIF(VALUES(winner_contact), ''), winner_contact),
                winner_phone = COALESCE(NULLIF(VALUES(winner_phone), ''), winner_phone),
                bid_amount = COALESCE(VALUES(bid_amount), bid_amount),
                agency_service_fee_amount = COALESCE(VALUES(agency_service_fee_amount), agency_service_fee_amount),
                bid_content = COALESCE(NULLIF(VALUES(bid_content), ''), bid_content),
                project_location = COALESCE(NULLIF(VALUES(project_location), ''), project_location),
                tender_file = COALESCE(NULLIF(VALUES(tender_file), ''), tender_file)
        """
        try:
            with conn.cursor() as cur:
                for i, r in enumerate(records):
                    h = _url_hash(r["detail_url"])
                    # 获取对应的详情数据
                    detail = detail_data[i] if detail_data and i < len(detail_data) else {}
                    
                    affected = cur.execute(sql, (
                        r["title"],
                        r.get("project_no") or r.get("purchase_no"),
                        r.get("publish_date"),
                        r["detail_url"],
                        h,
                        _fit_varchar(detail.get("bidder"), 200),
                        _fit_varchar(detail.get("bidder_contact"), 200),
                        _fit_varchar(detail.get("bidder_phone"), 50),
                        _fit_varchar(detail.get("agency"), 200),
                        _fit_varchar(_normalize_agency_contact(detail.get("agency_contact") or detail.get("agent_contat")), 200),
                        _fit_varchar(detail.get("agency_phone"), 50),
                        _fit_varchar(detail.get("winner"), 200),
                        _fit_varchar(detail.get("winner_contact"), 50),
                        _fit_varchar(detail.get("winner_phone"), 50),
                        detail.get("bid_amount"),
                        detail.get("agency_service_fee_amount"),
                        detail.get("bid_content"),
                        _fit_varchar(detail.get("project_location"), 200),
                        _fit_varchar(detail.get("tender_file"), 500),
                    ))
                    if affected:
                        inserted += 1
                    else:
                        skipped += 1
            conn.commit()
        finally:
            conn.close()
        return inserted, skipped
    
    else:
        logger.warning(f"未知 channel_id: {channel_id}，跳过")
        return 0, 0
    return inserted, skipped

# ─────────────────────────────────────────────
#  ④ 解析逻辑
# ─────────────────────────────────────────────
def parse_list_page(html: str) -> tuple[list[dict], bool]:
    """
    解析列表页 HTML。
    返回 (records, has_next)
    """
    soup = BeautifulSoup(html, "html.parser")
    records = []

    # 新站点结构（PC+移动）:
    # 1) PC:   <div id="allList"><ul><li onclick="window.open('/jyxxzcgg/xxx?...')">...</li></ul></div>
    # 2) 移动: <div class="detailListContent"><ul><li onclick="window.open('/jyxxzcgg/xxx')">...</li></ul></div>
    rows = soup.select("#allList ul li, .detailListContent ul li")
    if rows:
        for row in rows:
            onclick = row.get("onclick", "")
            m = re.search(r"window\.open\('([^']+)'\)", onclick)
            if not m:
                continue
            href = m.group(1).strip()
            if not href:
                continue

            # 优先按可见文本抽取（兼容 PC/移动）
            title = ""
            project_no = ""
            date_text = ""
            span_texts = []

            # 标题通常在 p.color3 的最后一个 span
            title_node = row.select_one("p.color3 span:last-child") or row.select_one(".cs-span2")
            if title_node:
                title = title_node.get_text(" ", strip=True)

            # 项目编号、发布时间通常在 p.color7 文本中
            for p in row.select("p"):
                txt = p.get_text(" ", strip=True).replace("\xa0", "").strip()
                if (not project_no) and ("项目编号" in txt):
                    mm = re.search(r"项目编号[：:]\s*(.+)$", txt)
                    candidate = mm.group(1).strip() if mm else txt.replace("项目编号", "").strip("：: ")
                    project_no = _normalize_project_no(candidate) or project_no
                if (not date_text) and ("发布时间" in txt or "发布日期" in txt):
                    mm = re.search(r"(发布时间|发布日期)[：:]\s*(.+)$", txt)
                    date_text = mm.group(2).strip() if mm else txt

            # 兜底：老 PC 列表常见 span 排列 [空, 标题, 编号, 日期]
            if not title or not date_text:
                span_texts = [
                    s.get_text(" ", strip=True).replace("\xa0", "").strip()
                    for s in row.select("span")
                ]
                span_texts = [t for t in span_texts if t]
                if span_texts:
                    title = title or span_texts[0]
                if len(span_texts) >= 2:
                    project_no = project_no or (_normalize_project_no(span_texts[1]) or "")
                if len(span_texts) >= 3:
                    date_text = date_text or span_texts[2]

            if not project_no:
                row_text = " ".join(
                    [row.get_text(" ", strip=True)] + span_texts +
                    [p.get_text(" ", strip=True) for p in row.select("p")]
                )
                project_no = _extract_project_no_from_text(row_text) or ""

            if not title:
                continue

            # 补全 URL
            if href.startswith("http"):
                detail_url = href
            elif href.startswith("/"):
                detail_url = BASE_URL + href
            else:
                detail_url = BASE_URL + "/" + href

            # 解析日期
            pub_date = None
            m_date = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", date_text)
            if m_date:
                try:
                    pub_date = datetime.strptime(
                        m_date.group(1).replace("/", "-"), "%Y-%m-%d"
                    ).date()
                except ValueError:
                    pass

            records.append({
                "title": title,
                "project_no": project_no or "",
                "publish_date": pub_date,
                "detail_url": detail_url,
            })

        # 基于页面内分页脚本参数判断是否有下一页
        # 例如: var showCount = 99957; var limit = 10; ,curr: 1
        m_count = re.search(r"var\s+(?:showCount|count)\s*=\s*(\d+)", html)
        m_limit = re.search(r"var\s+limit\s*=\s*(\d+)", html)
        m_curr = re.search(r",curr:\s*(\d+)", html)
        if m_count and m_limit and m_curr:
            total_count = int(m_count.group(1))
            page_size = int(m_limit.group(1))
            curr_page = int(m_curr.group(1))
            has_next = curr_page * page_size < total_count
        else:
            # 保守策略：交给外层的空页/重复页逻辑终止
            has_next = bool(records)
        return records, has_next

    # 兼容旧结构：table 或 li+a
    rows = soup.select("table tbody tr, table.list-table tr")
    if not rows:
        rows = soup.select("ul.search-list li, ul.result-list li")

    for row in rows:
        cells = row.select("td")
        if len(cells) >= 2:
            # Table 布局：td[0]=标题链接, td[1]=项目编号, td[2]=日期
            a_tag = cells[0].find("a")
            if not a_tag:
                continue
            title      = a_tag.get_text(strip=True)
            href       = a_tag.get("href", "")
            project_no = _normalize_project_no(cells[1].get_text(" ", strip=True) if len(cells) > 1 else None)
            date_text  = cells[2].get_text(strip=True) if len(cells) > 2 else None
        else:
            # Li 布局：标题 + 日期
            a_tag = row.find("a")
            if not a_tag:
                continue
            title      = a_tag.get_text(strip=True)
            href       = a_tag.get("href", "")
            project_no = None
            date_span  = row.find("span", class_=re.compile(r"date|time"))
            date_text  = date_span.get_text(strip=True) if date_span else None

        if not title or not href:
            continue

        # 补全 URL
        if href.startswith("http"):
            detail_url = href
        elif href.startswith("/"):
            detail_url = BASE_URL + href
        else:
            detail_url = BASE_URL + "/" + href

        # 解析日期
        pub_date = None
        if date_text:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", date_text)
            if m:
                try:
                    pub_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                except ValueError:
                    pass

        if not project_no:
            project_no = _extract_project_no_from_text(row.get_text(" ", strip=True))

        records.append({
            "title":        title,
            "project_no":   project_no or "",
            "publish_date": pub_date,
            "detail_url":   detail_url,
        })

    # 判断是否有下一页
    has_next = bool(soup.select(
        "a.next, a[rel='next'], .pager .next:not(.disabled), li.next:not(.disabled) a"
    ))
    # 兜底：如果本页有正常条目且未出现明确的"没有更多"提示
    no_more = soup.find(string=re.compile(r"没有更多|暂无数据|共\s*0\s*条"))
    if no_more:
        has_next = False

    return records, has_next


def parse_detail_page(html: str) -> dict:
    """
    解析中标结果公告详情页，提取详细字段。
    优先使用 CSS 选择器解析网页标签，正则做兜底。
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {}
    full_text = soup.get_text()
    full_text_compact = re.sub(r"\s+", " ", full_text)

    def _pick(patterns: list[str]) -> str | None:
        for p in patterns:
            m = re.search(p, full_text_compact, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _extract_from_block(block: str, patterns: list[str]) -> str | None:
        if not block:
            return None
        for p in patterns:
            m = re.search(p, block, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _extract_phone(text: str | None) -> str | None:
        """提取常见电话格式：座机/手机号/400，兼容空格与分机。"""
        if not text:
            return None
        t = str(text)
        # 手机号
        m_mobile = re.search(r"(1\d{10})", t)
        if m_mobile:
            return m_mobile.group(1)

        # 400电话
        m_400 = re.search(r"(400)[-\s]?(\d{3})[-\s]?(\d{4})", t)
        if m_400:
            return f"{m_400.group(1)}-{m_400.group(2)}-{m_400.group(3)}"

        # 本地固话（无区号）：7-8位
        m_local = re.search(r"(?<!\d)(\d{7,8})(?!\d)", t)
        if m_local:
            return m_local.group(1)

        # 连续座机号码（无分隔符）：如 02135968033 / 01012345678
        m_tel_compact = re.search(r"(?<!\d)(0\d{2})(\d{8})(?!\d)", t)
        if m_tel_compact:
            return f"{m_tel_compact.group(1)}-{m_tel_compact.group(2)}"
        m_tel_compact2 = re.search(r"(?<!\d)(0\d{3})(\d{7,8})(?!\d)", t)
        if m_tel_compact2:
            return f"{m_tel_compact2.group(1)}-{m_tel_compact2.group(2)}"

        # 座机（分机可存在，但仅保留主号码）
        m_tel = re.search(r"(0\d{2,3})[-\s]?(\d{7,8})(?:[-\s]*(?:转|ext\.?)?\s*\d{1,6})?", t, flags=re.IGNORECASE)
        if m_tel:
            return f"{m_tel.group(1)}-{m_tel.group(2)}"
        return None
    
    # ========================================
    # 辅助函数：CSS选择器提取 + 正则兜底
    # ========================================
    def extract_by_label(label_text: str, field_name: str, is_phone=False):
        """根据标签文字找到相邻的值"""
        # 1. 尝试用 CSS 选择器找到包含 label_text 的标签
        # 常见结构：<tr><td>招标人</td><td>xxx</td></tr> 或 <span class="label">招标人</span><span>xxx</span>
        selectors = [
            f'td:contains("{label_text}") + td',           # table tr 结构
            f'tr:contains("{label_text}") td:nth-child(2)',  # tr 结构
            f'span:contains("{label_text}") + span',       # span 相邻
            f'p:contains("{label_text}")',                # p 段落
            f'div:contains("{label_text}")',              # div 块
        ]
        
        for sel in selectors:
            try:
                elems = soup.select(sel)
                for elem in elems:
                    text = elem.get_text(strip=True)
                    if text and text != label_text:
                        if is_phone:
                            # 提取电话号码
                            phone_match = re.search(r'[\d\-]{7,15}', text)
                            if phone_match:
                                result[field_name] = phone_match.group()
                                return
                        else:
                            result[field_name] = text
                            return
            except Exception:
                continue
        
        # 2. 正则兜底
        patterns = [
            f"{label_text}[：:]\\s*([^\\n\\r]{{2,100}})",
            f"{label_text}\\s*[:：]\\s*([\\u4e00-\\u9fa5a-zA-Z0-9]{{2,50}})",
        ]
        for p in patterns:
            m = re.search(p, full_text)
            if m:
                val = m.group(1).strip()
                if is_phone:
                    phone = re.search(r'[\d\-]{7,15}', val)
                    if phone:
                        result[field_name] = phone.group()
                else:
                    result[field_name] = val
                return
    
    # ========================================
    # 0. 新模板键值对（.table_1_ul）优先提取
    # ========================================
    table_kv = {}
    for ul in soup.select(".table_1_ul, .table_1_ulContent"):
        lis = ul.select(":scope > li")
        if len(lis) < 2:
            continue
        i = 0
        while i + 1 < len(lis):
            key = lis[i].get_text(" ", strip=True).strip("：: ")
            val = lis[i + 1].get_text(" ", strip=True).replace("\xa0", " ").strip()
            if key and val:
                table_kv[key] = val
            i += 2

    if "采购代理机构名称" in table_kv and "agency" not in result:
        result["agency"] = table_kv["采购代理机构名称"]
    if "中标（成交）供应商名称" in table_kv and "winner" not in result:
        result["winner"] = table_kv["中标（成交）供应商名称"]
    if "公告内容" in table_kv and "bid_content" not in result:
        result["bid_content"] = table_kv["公告内容"]
    for fee_key in ("代理服务收费金额（元）", "代理服务收费金额(元)", "代理服务收费金额"):
        if fee_key in table_kv and "agency_service_fee_amount" not in result:
            m_fee = re.search(r"([\d,]+(?:\.\d+)?)", table_kv[fee_key].replace(",", ""))
            if m_fee:
                try:
                    result["agency_service_fee_amount"] = float(m_fee.group(1))
                except ValueError:
                    pass
            break

    # 采购公告常见结构：1.采购人信息 / 2.采购代理机构信息 / 3.项目联系方式
    sec1 = _pick([r"(?:1[\.、]\s*采购人信息)(.*?)(?:(?:2[\.、]\s*采购代理机构信息)|(?:3[\.、]\s*项目联系方式)|$)"])
    sec2 = _pick([r"(?:2[\.、]\s*采购代理机构信息)(.*?)(?:(?:3[\.、]\s*项目联系方式)|$)"])
    sec3 = _pick([r"(?:3[\.、]\s*项目联系方式)(.*)$"])

    if sec1:
        if "bidder" not in result:
            v = _extract_from_block(sec1, [r"(?:名称|名\s*称)[：:]\s*([^；;，,。]{2,200})"])
            if v:
                result["bidder"] = v
        if "bidder_contact" not in result:
            v = _extract_from_block(sec1, [r"(?:地址|地\s*址)[：:]\s*([^；;，,。]{2,240})"])
            if v:
                result["bidder_contact"] = v
        if "bidder_phone" not in result:
            v = _extract_from_block(sec1, [r"(?:联系方式|联系电话|电话)[：:]\s*([^；;，,。]{7,40})"])
            phone = _extract_phone(v)
            if phone:
                result["bidder_phone"] = phone

    if sec2:
        if "agency" not in result:
            v = _extract_from_block(sec2, [r"(?:名称|名\s*称)[：:]\s*([^；;，,。]{2,200})"])
            if v:
                result["agency"] = v
        if "agency_contact" not in result:
            v = _extract_from_block(sec2, [r"(?:地址|地\s*址)[：:]\s*([^；;，,。]{2,240})"])
            if v:
                result["agency_contact"] = v
        if "agency_phone" not in result:
            v = _extract_from_block(sec2, [r"(?:联系方式|联系电话|电话)[：:]\s*([^；;，,。]{7,40})"])
            phone = _extract_phone(v)
            if phone:
                result["agency_phone"] = phone

    if sec3:
        if "project_contact" not in result:
            v = _extract_from_block(sec3, [r"(?:项目联系人)[：:]\s*([^；;，,。]{2,80})"])
            if v:
                result["project_contact"] = v
        if "project_phone" not in result:
            v = _extract_from_block(sec3, [r"(?:电话|联系方式)[：:]\s*([^；;，,。]{7,40})"])
            phone = _extract_phone(v)
            if phone:
                result["project_phone"] = phone

    # 从“中标（成交）信息”表格中提取中标人/金额
    for tbl in soup.select("table"):
        rows = tbl.select("tr")
        if len(rows) < 2:
            continue
        headers = [c.get_text(" ", strip=True) for c in rows[0].select("th,td")]
        if not headers:
            continue
        winner_idx = amount_idx = None
        for idx, h in enumerate(headers):
            if winner_idx is None and ("中标供应商名称" in h or "成交供应商名称" in h or "供应商名称" in h):
                winner_idx = idx
            if amount_idx is None and ("中标（成交金额）" in h or "中标金额" in h or "成交金额" in h):
                amount_idx = idx
        if winner_idx is None and amount_idx is None:
            continue
        cells = [c.get_text(" ", strip=True) for c in rows[1].select("th,td")]
        if winner_idx is not None and winner_idx < len(cells) and not result.get("winner"):
            v = cells[winner_idx].strip()
            if v:
                result["winner"] = v
        if amount_idx is not None and amount_idx < len(cells) and "bid_amount" not in result:
            amt_txt = cells[amount_idx].replace(",", "")
            m_amt = re.search(r"(\d+(?:\.\d+)?)", amt_txt)
            if m_amt:
                try:
                    result["bid_amount"] = float(m_amt.group(1))
                except ValueError:
                    pass
        if result.get("winner") and "bid_amount" in result:
            break

    # ========================================
    # 1. 招标人（采购单位）
    # ========================================
    # CSS选择器尝试：查找表格中包含"招标人"的行
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if '招标人' in label or '采购人' in label or '单位名称' in label:
                val = cells[1].get_text(strip=True)
                if val:
                    result["bidder"] = val
                    break
    
    if "bidder" not in result:
        extract_by_label("招标人", "bidder")
    
    # ========================================
    # 2. 招标人地址（bidder_contact 字段复用为地址）
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if ('地址' in label or '联系地址' in label or '通讯地址' in label) and ('招标人' in label or '采购人' in label):
                result["bidder_contact"] = cells[1].get_text(strip=True)
                break
    
    if "bidder_contact" not in result:
        extract_by_label("采购人地址", "bidder_contact")
    if "bidder_contact" not in result:
        extract_by_label("联系地址", "bidder_contact")
    
    # ========================================
    # 3. 招标人联系方式（电话）
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if ('电话' in label or '联系方式' in label) and '招标人' in label:
                result["bidder_phone"] = cells[1].get_text(strip=True)
                break
    
    if "bidder_phone" not in result:
        extract_by_label("采购人联系方式", "bidder_phone", is_phone=True)
    if "bidder_phone" not in result:
        extract_by_label("联系电话", "bidder_phone", is_phone=True)
    
    # ========================================
    # 4. 招标代理机构
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if '代理' in label and ('机构' in label or '公司' in label):
                result["agency"] = cells[1].get_text(strip=True)
                break
    
    if "agency" not in result:
        extract_by_label("代理机构", "agency")
    
    # ========================================
    # 5. 招标代理机构地址（优先）/联系人（兜底）
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if ('地址' in label or '联系地址' in label or '通讯地址' in label) and '代理' in label:
                result["agency_contact"] = cells[1].get_text(strip=True)
                break
    
    if "agency_contact" not in result:
        # 兼容旧模板：如果没有地址，再退化到联系人
        extract_by_label("代理机构联系人", "agency_contact")
    
    # ========================================
    # 6. 招标代理机构联系方式
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if ('电话' in label or '联系方式' in label) and '代理' in label:
                result["agency_phone"] = cells[1].get_text(strip=True)
                break
    
    if "agency_phone" not in result:
        extract_by_label("代理机构电话", "agency_phone", is_phone=True)
    
    # ========================================
    # 7. 中标人
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if '中标人' in label or '成交供应商' in label or '供应商' in label:
                result["winner"] = cells[1].get_text(strip=True)
                break
    
    if "winner" not in result:
        extract_by_label("中标人", "winner")
    
    # ========================================
    # 8. 中标人联系人
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if '联系人' in label and '中标' in label:
                result["winner_contact"] = cells[1].get_text(strip=True)
                break
    
    if "winner_contact" not in result:
        extract_by_label("中标人联系人", "winner_contact")
    
    # ========================================
    # 9. 中标人联系方式
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if ('电话' in label or '联系方式' in label) and '中标' in label:
                result["winner_phone"] = cells[1].get_text(strip=True)
                break
    
    if "winner_phone" not in result:
        extract_by_label("中标人电话", "winner_phone", is_phone=True)

    # ========================================
    # 9.5 项目联系方式（采购公告常用）
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            val = cells[1].get_text(strip=True)
            if not val:
                continue
            if "项目联系人" in label and "project_contact" not in result:
                result["project_contact"] = val
            if ("电话" in label or "联系方式" in label) and ("项目" in label or "联系方式" in label) and "project_phone" not in result:
                m = re.search(r"[0-9\-（）()]{7,25}", val)
                if m:
                    result["project_phone"] = re.sub(r"[^\d\-]", "", m.group())
    
    # ========================================
    # 10. 中标金额
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if '金额' in label and ('中标' in label or '成交' in label or '投标' in label):
                text = cells[1].get_text(strip=True)
                amount_match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
                if amount_match:
                    try:
                        result["bid_amount"] = float(amount_match.group())
                        break
                    except ValueError:
                        pass
    
    if "bid_amount" not in result:
        # 正则兜底
        for p in [r"中标金额[：:]\s*([\d,]+\.?\d*)", r"成交金额[：:]\s*([\d,]+\.?\d*)"]:
            m = re.search(p, full_text)
            if m:
                try:
                    result["bid_amount"] = float(m.group(1).replace(",", ""))
                    break
                except ValueError:
                    pass
    
    # ========================================
    # 11. 招标内容
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if '招标' in label and '内容' in label:
                result["bid_content"] = cells[1].get_text(strip=True)
                break
    
    if "bid_content" not in result:
        extract_by_label("招标内容", "bid_content")
    
    # ========================================
    # 12. 项目地点
    # ========================================
    for tr in soup.select('table tr'):
        cells = tr.select('td')
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if '地点' in label and ('项目' in label or '建设' in label):
                result["project_location"] = cells[1].get_text(strip=True)
                break
    
    if "project_location" not in result:
        extract_by_label("项目地点", "project_location")
    
    # ========================================
    # 13. 招标文件（PDF链接）
    # ========================================
    # 常见结构：<a href="xxx.pdf">招标文件</a>
    tender_link = soup.select_one('a:contains("招标文件"), a[href*="招标文件"], a[href$=".pdf"]')
    if not tender_link:
        # 尝试查找包含"招标文件"文字的链接
        for a in soup.select('a[href]'):
            if '招标' in a.get_text() or '文件' in a.get_text():
                tender_link = a
                break
    
    if tender_link:
        href = tender_link.get("href", "")
        if href.startswith("http"):
            result["tender_file"] = href
        elif href.startswith("/"):
            result["tender_file"] = BASE_URL + href
    
    # ========================================
    # 14. 通用正则兜底（适配新详情模板）
    # ========================================
    if "bidder" not in result:
        v = _pick([r"(?:采购人(?:名称)?|招标人(?:名称)?)[：:]\s*([^；。,\n]{2,80})"])
        if v:
            result["bidder"] = v
    if "bidder" not in result:
        v = _pick([r"1\.采购人信息.*?名称[：:]\s*([^；。,\n]{2,80})"])
        if v:
            result["bidder"] = v
    if "bidder_contact" not in result:
        v = _pick([r"(?:采购人|招标人).{0,20}(?:地址|联系地址|通讯地址)[：:]\s*([^；。,\n]{2,120})"])
        if v:
            result["bidder_contact"] = v
    if "bidder_phone" not in result:
        v = _pick([r"(?:采购人|招标人).{0,20}(?:电话|联系方式)[：:]\s*([0-9\-]{7,20})"])
        if v:
            result["bidder_phone"] = v
    if "bidder_phone" not in result:
        v = _pick([r"1\.采购人信息.*?(?:联系方式|电话)[：:]\s*([0-9\-（）()]{7,25})"])
        if v:
            result["bidder_phone"] = re.sub(r"[^\d\-]", "", v)
    if "agency" not in result:
        v = _pick([r"(?:采购代理机构(?:名称)?|招标代理机构(?:名称)?)[：:]\s*([^；。,\n]{2,100})"])
        if v:
            result["agency"] = v
    if "agency_contact" not in result:
        v = _pick([r"(?:代理机构|采购代理).{0,20}(?:地址|联系地址|通讯地址)[：:]\s*([^；。,\n]{2,120})"])
        if v:
            result["agency_contact"] = v
    if "agency_contact" not in result:
        v = _pick([r"(?:代理机构|采购代理).{0,20}联系人[：:]\s*([^；。,\n]{2,30})"])
        if v:
            result["agency_contact"] = v
    if "agency_phone" not in result:
        v = _pick([r"(?:代理机构|采购代理).{0,20}(?:电话|联系方式)[：:]\s*([0-9\-]{7,20})"])
        if v:
            result["agency_phone"] = v
    if "agency_phone" not in result:
        v = _pick([r"2\.采购代理机构信息.*?(?:联系方式|电话)[：:]\s*([0-9\-（）()]{7,25})"])
        if v:
            result["agency_phone"] = re.sub(r"[^\d\-]", "", v)
    if "winner" not in result:
        v = _pick([
            r"(?:中标（?成交）?供应商(?:名称)?|中标人(?:名称)?|成交供应商(?:名称)?)[：:]\s*([^；。,\n]{2,120})",
            r"供应商(?:名称)?[：:]\s*([^；。,\n]{2,120})",
        ])
        if v:
            result["winner"] = v
    if "winner_contact" not in result:
        v = _pick([r"(?:中标（?成交）?供应商|中标人|成交供应商).{0,20}联系人[：:]\s*([^；。,\n]{2,30})"])
        if v:
            result["winner_contact"] = v
    if "project_contact" not in result:
        v = _pick([r"3\.项目联系方式.*?项目联系人[：:]\s*([^；。,\n]{2,40})"])
        if v:
            result["project_contact"] = v
    if "project_phone" not in result:
        v = _pick([r"3\.项目联系方式.*?(?:电话|联系方式)[：:]\s*([^；。,\n]{7,40})"])
        phone = _extract_phone(v)
        if phone:
            result["project_phone"] = phone
    if "project_phone" not in result and "project_contact" in result:
        # 某些页面把联系人和电话写在同一行：仅在联系人附近窗口兜底，避免抓到页面公共电话
        anchor = str(result.get("project_contact") or "")
        pos = full_text_compact.find(anchor)
        window = None
        if pos >= 0:
            left = max(0, pos - 40)
            right = min(len(full_text_compact), pos + 160)
            window = full_text_compact[left:right]
        phone = _extract_phone(window or "")
        if phone:
            result["project_phone"] = phone
    if "winner_phone" not in result:
        v = _pick([r"(?:中标（?成交）?供应商|中标人|成交供应商).{0,20}(?:电话|联系方式)[：:]\s*([0-9\-]{7,20})"])
        if v:
            result["winner_phone"] = v
    if "bid_amount" not in result:
        v = _pick([
            r"(?:中标（?成交）?金额|中标金额|成交金额)[：:]\s*[¥￥]?\s*([\d,]+(?:\.\d+)?)",
            r"报价[：:]\s*[¥￥]?\s*([\d,]+(?:\.\d+)?)",
        ])
        if v:
            try:
                result["bid_amount"] = float(v.replace(",", ""))
            except ValueError:
                pass
    if "agency_service_fee_amount" not in result:
        v = _pick([
            r"(?:代理服务收费金额(?:（元）|\(元\))?|代理服务费金额)[：:]\s*[¥￥]?\s*([\d,]+(?:\.\d+)?)",
            r"代理服务收费金额（元）[：:]\s*([\d,]+(?:\.\d+)?)",
        ])
        if v:
            try:
                result["agency_service_fee_amount"] = float(v.replace(",", ""))
            except ValueError:
                pass

    # bidder 字段后处理：统一从 bidder 原始文本拆分
    # - bidder: 仅公司名称
    # - bidder_contact: 地址
    # - bidder_phone: 联系方式
    if "bidder" in result:
        bidder_name, bidder_addr, bidder_phone = _split_bidder_fields(result.get("bidder"))
        if bidder_name:
            result["bidder"] = bidder_name
        if bidder_addr:
            result["bidder_contact"] = bidder_addr
        if bidder_phone:
            result["bidder_phone"] = bidder_phone

    # agency 字段后处理：名称/地址/联系方式拆分
    if "agency" in result:
        agency_name, agency_addr, agency_phone = _split_agency_fields(result.get("agency"))
        if agency_name:
            result["agency"] = agency_name
        if agency_addr:
            result["agency_contact"] = agency_addr
        if agency_phone:
            result["agency_phone"] = agency_phone

    # agency_contact 兼容“地址/联系人姓名”两种语义
    if "agency_contact" in result:
        normalized = _normalize_agency_contact(result.get("agency_contact"))
        if normalized:
            result["agency_contact"] = normalized
    if "project_contact" in result:
        cleaned = _clean_contact_name(result.get("project_contact"))
        if cleaned:
            result["project_contact"] = cleaned

    # 长度保护，避免异常长文本导致入库报错
    result["bidder"] = _fit_varchar(result.get("bidder"), 200)
    result["bidder_contact"] = _fit_varchar(result.get("bidder_contact"), 200)
    result["bidder_phone"] = _fit_varchar(result.get("bidder_phone"), 50)
    result["agency"] = _fit_varchar(result.get("agency"), 200)
    result["agency_contact"] = _fit_varchar(result.get("agency_contact"), 200)
    result["agency_phone"] = _fit_varchar(result.get("agency_phone"), 50)
    result["project_contact"] = _fit_varchar(result.get("project_contact"), 100)
    result["project_phone"] = _fit_varchar(result.get("project_phone"), 50)

    return result


def fetch_detail(url: str, session: requests.Session) -> dict:
    """抓取并解析详情页"""
    normalized = _normalize_detail_url(url)
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = session.get(normalized, headers=_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            if not resp.text.strip():
                raise requests.RequestException("empty detail response body")
            detail = parse_detail_page(resp.text)

            # 兜底：若当前详情页信息缺失，尝试跳转“信息来源URL”抓原始公告页再解析
            weak = not (detail.get("bidder") or detail.get("agency") or detail.get("project_contact"))
            if weak:
                m_src = re.search(
                    r"(https?://www\.zfcg\.sh\.gov\.cn/luban/detail\?[^\s\"'<>]+)",
                    html_lib.unescape(resp.text),
                    flags=re.IGNORECASE,
                )
                if m_src:
                    src_url = m_src.group(1).replace("&amp;", "&")
                    try:
                        src_resp = session.get(src_url, headers=_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
                        src_resp.raise_for_status()
                        src_resp.encoding = src_resp.apparent_encoding or "utf-8"
                        src_detail = parse_detail_page(src_resp.text)
                        # 仅在原始页有更完整信息时覆盖
                        for k in (
                            "bidder", "bidder_contact", "bidder_phone",
                            "agency", "agency_contact", "agency_phone",
                            "project_contact", "project_phone",
                        ):
                            if (not detail.get(k)) and src_detail.get(k):
                                detail[k] = src_detail.get(k)
                    except Exception:
                        pass

            return detail
        except requests.RequestException as e:
            if attempt >= REQUEST_RETRIES:
                logger.warning(f"详情页抓取失败: {normalized}, 错误: {e}")
                return {}
            time.sleep(1.2 * attempt)
    return {}


# ─────────────────────────────────────────────
#  ⑤ 爬取主逻辑
# ─────────────────────────────────────────────
def fetch_page(page_no: int, session: requests.Session, channel_id: int = None) -> str | None:
    """抓取第 page_no 页，返回 HTML 或 None"""
    if channel_id is None:
        channel_id = CHANNEL_IDS[0]
    
    cext = _make_cext(page_no)
    # 按站点真实路由分页，避免 pageNo 参数被忽略导致始终返回第一页
    if channel_id == 37:
        # 采购公告列表页路由: /jyxxzcgg, /jyxxzcgg_2.jhtml ...
        url = f"{BASE_URL}/jyxxzcgg" if page_no == 1 else f"{BASE_URL}/jyxxzcgg_{page_no}.jhtml"
        params = {
            "cExt": cext,
            "isIndex": "",
        }
    elif channel_id == 38:
        # 中标结果列表页路由: /search/queryContents.jhtml, /search/queryContents_2.jhtml ...
        url = SEARCH_URL if page_no == 1 else f"{BASE_URL}/search/queryContents_{page_no}.jhtml"
        params = {
            "title":     "",
            "channelId": 38,
            "origin":    "",
            "inDates":   IN_DATES,
            "ext":       "",
            "timeBegin": "",
            "timeEnd":   "",
            "ext1":      "",
            "ext2":      "",
            "cExt":      cext,
        }
    else:
        url = SEARCH_URL
        params = {
            "title":     "",
            "channelId": channel_id,
            "origin":    "",
            "inDates":   IN_DATES,
            "ext":       "",
            "timeBegin": "",
            "timeEnd":   "",
            "ext1":      "",
            "ext2":      "",
            "pageNo":    page_no,
            "cExt":      cext,
        }
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = session.get(
                url,
                params=params,
                headers=_headers(),
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            if attempt >= REQUEST_RETRIES:
                logger.warning(f"第 {page_no} 页请求失败: {e}")
                return None
            time.sleep(1.2 * attempt)
    return None

def _crawl_one_page(
    channel_id: int, page_no: int, session: requests.Session, target_day: date
) -> tuple[bool, bool, int, int, bool]:
    """
    抓取单页并入库。
    返回:
      ok:       页面请求和解析是否成功
      has_next: 是否存在下一页
      inserted: 新增条数
      skipped:  已存在/未更新条数
      all_older_than_target: 当前页日期是否均早于目标日期（可提前终止翻页）
    """
    channel_name = CHANNEL_NAMES.get(channel_id, f"频道{channel_id}")
    logger.info(f"  [{channel_name}] 第 {page_no} 页 ...")
    html = fetch_page(page_no, session, channel_id)
    if not html:
        logger.warning(f"  └─ 第 {page_no} 页请求失败，跳过")
        return False, True, 0, 0, False

    records, has_next = parse_list_page(html)
    logger.info(f"  └─ 解析到 {len(records)} 条公告")
    if not records:
        return True, has_next, 0, 0, False

    # 仅保留目标日期（默认昨天）数据，其他日期跳过
    filtered_records = []
    dated_records = 0
    older_records = 0
    for rec in records:
        pub = rec.get("publish_date")
        if isinstance(pub, date):
            dated_records += 1
            if pub == target_day:
                filtered_records.append(rec)
            elif pub < target_day:
                older_records += 1
        else:
            # 无日期记录默认跳过，避免脏数据进入增量表
            continue

    if len(filtered_records) != len(records):
        logger.info(f"  └─ 目标日期过滤后保留 {len(filtered_records)} 条（目标: {target_day}）")

    all_older_than_target = dated_records > 0 and older_records == dated_records
    if not filtered_records:
        return True, has_next, 0, 0, all_older_than_target

    detail_data = None
    if channel_id in (37, 38):
        logger.info(f"  └─ 正在抓取详情页...")
        detail_data = []
        for i, rec in enumerate(filtered_records):
            logger.info(f"    └─ [{i+1}/{len(filtered_records)}] {rec['title'][:30]}...")
            detail = fetch_detail(rec["detail_url"], session)
            detail_data.append(detail)
            _sleep()

    ins, skp = save_records(filtered_records, channel_id, detail_data)
    logger.info(f"  └─ 入库 {ins} 条，跳过(已存在) {skp} 条")
    return True, has_next, ins, skp, all_older_than_target


def crawl():
    """主爬取函数，每次调用完成一轮抓取"""
    target_day = _target_date()
    logger.info(
        f"===== 开始抓取 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，目标日期: {target_day} ====="
    )
    session = requests.Session()

    total_inserted = total_skipped = 0
    first_page_status: dict[int, tuple[bool, bool]] = {}

    # 第一阶段：先抓两个频道第一页，先确保数据正确
    logger.info(">>> 第一阶段：先抓取各频道第 1 页（数据校验阶段）")
    for channel_id in CHANNEL_IDS:
        channel_name = CHANNEL_NAMES.get(channel_id, f"频道{channel_id}")
        logger.info(f"\n>>> 正在抓取频道: {channel_name} (channelId={channel_id})")
        ok, has_next, ins, skp, all_older = _crawl_one_page(channel_id, 1, session, target_day)
        total_inserted += ins
        total_skipped += skp
        first_page_status[channel_id] = (ok, has_next and (not all_older))
        _sleep()

    if FIRST_PAGE_VALIDATE_GATE:
        bad_channels = [
            CHANNEL_NAMES.get(cid, str(cid))
            for cid, (ok, _) in first_page_status.items()
            if not ok
        ]
        if bad_channels:
            logger.error(f"第一页校验失败，暂停后续翻页: {', '.join(bad_channels)}")
            logger.success(
                f"===== 本轮完成：新增 {total_inserted} 条，跳过 {total_skipped} 条 ====="
            )
            return

    # 第二阶段：第一页通过后，再循环抓取后续页
    logger.info(">>> 第二阶段：开始循环抓取后续分页")
    for channel_id in CHANNEL_IDS:
        channel_name = CHANNEL_NAMES.get(channel_id, f"频道{channel_id}")
        ok_first, has_next_first = first_page_status.get(channel_id, (False, False))
        if not ok_first:
            logger.warning(f">>> {channel_name} 跳过后续翻页（第一页失败）")
            continue
        if not has_next_first:
            logger.info(f">>> {channel_name} 第 1 页已是最后一页")
            continue

        empty_pages = 0
        for page_no in range(2, MAX_PAGES + 1):
            ok, has_next, ins, skp, all_older = _crawl_one_page(channel_id, page_no, session, target_day)
            if not ok:
                empty_pages += 1
                if empty_pages >= 3:
                    logger.error(f"  └─ 连续 3 页失败，终止 {channel_name} 频道")
                    break
                _sleep()
                continue

            total_inserted += ins
            total_skipped += skp
            if ins == 0 and skp == 0:
                empty_pages += 1
                logger.info(f"  └─ 空页（连续第 {empty_pages} 次）")
                if empty_pages >= 2:
                    logger.info(f"  └─ 连续 2 页无数据，停止 {channel_name} 翻页")
                    break
            else:
                empty_pages = 0
                if INCREMENTAL_STOP_ON_DUP_PAGE and skp > 0 and ins == 0:
                    logger.info(f"  └─ 本页全部为已存在数据，停止翻页（增量模式）")
                    break

            if all_older:
                logger.info(f"  └─ 本页日期均早于目标日期 {target_day}，停止翻页")
                break

            if not has_next:
                logger.info(f"  └─ 已到最后一页，停止翻页")
                break
            _sleep()

        logger.info(f">>> {channel_name} 频道抓取完成")

    logger.success(
        f"===== 本轮完成：新增 {total_inserted} 条，跳过 {total_skipped} 条 ====="
    )

# ─────────────────────────────────────────────
#  ⑥ 定时调度（每天 06:00 执行一次）
# ─────────────────────────────────────────────
def run_scheduler():
    logger.info(f"调度器启动，每天 {SCHEDULE_TIME} 执行一次增量抓取（昨天数据）")
    if RUN_ON_START:
        logger.info("启动即执行一次 ...")
        crawl()
    schedule.every().day.at(SCHEDULE_TIME).do(crawl)
    while True:
        schedule.run_pending()
        time.sleep(30)

# ─────────────────────────────────────────────
#  ⑦ 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 配置日志格式
    logger.add(
        "shggzy_crawler_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        encoding="utf-8",
        level="INFO",
    )

    # 初始化数据库
    init_db()

    # 启动（立即运行一次 + 每天定时）
    run_scheduler()
