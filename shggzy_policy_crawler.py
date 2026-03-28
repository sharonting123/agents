"""
上海市公共资源交易中心 — 国家政策法规（招标相关政策）爬虫

目标列表: https://www.shggzy.com/gjzcfg.jhtml
列表分页: 第 1 页 gjzcfg.jhtml，第 n 页 gjzcfg_{n}.jhtml（与站点 layui 分页一致）
详情路由: /gjzcfg/{数字ID}

逻辑与请求习惯对齐 shggzy_crawler.py（随机 UA、间隔、重试、MySQL 去重入库）。

依赖:
    pip install requests beautifulsoup4 pymysql loguru fake-useragent

用法:
    python shggzy_policy_crawler.py              # 默认最多抓第 10 页列表后停止，详情入库
    python shggzy_policy_crawler.py --max-pages 500  # 放宽页数上限
    python shggzy_policy_crawler.py --dry-run        # 只抓第 1 页列表+第 1 条详情，不入库
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pymysql
import requests
from bs4 import BeautifulSoup
from loguru import logger
from fake_useragent import UserAgent

# ─────────────────────────────────────────────
# 配置（与 shggzy_crawler.py 保持一致即可）
# ─────────────────────────────────────────────
MYSQL_CONFIG = {
    "host": "39.105.216.24",
    "port": 3306,
    "user": "ztt",
    "password": "Ti123456!",
    "database": "bidding",
    "charset": "utf8mb4",
}

BASE_URL = "https://www.shggzy.com"
POLICY_LIST_FIRST = f"{BASE_URL}/gjzcfg.jhtml"
# 频道标识：国家政策法规（gjzcfg），便于以后扩展上海汇编、中心制度等
POLICY_CHANNEL = "gjzcfg"

REQUEST_INTERVAL = (2, 4)
REQUEST_TIMEOUT = 25
REQUEST_RETRIES = 3
# 默认最多抓取列表页数（采集完第 N 页即停，不再翻页）
MAX_PAGES = 10

ua = UserAgent()
DESKTOP_FALLBACK_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


def _desktop_ua() -> str:
    try:
        candidate = ua.random
    except Exception:
        return DESKTOP_FALLBACK_UA
    lower = candidate.lower()
    if any(k in lower for k in ("mobile", "iphone", "android", "ipad")):
        return DESKTOP_FALLBACK_UA
    return candidate


def _headers() -> dict:
    return {
        "User-Agent": _desktop_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": BASE_URL + "/",
        "Connection": "keep-alive",
    }


def _sleep() -> None:
    time.sleep(random.uniform(*REQUEST_INTERVAL))


def _stable_url_for_hash(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    q.sort(key=lambda x: (x[0], x[1]))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(q, doseq=True), ""))


def _url_hash(url: str) -> str:
    return hashlib.md5(_stable_url_for_hash(url).encode()).hexdigest()


def _get_conn():
    return pymysql.connect(**MYSQL_CONFIG)


def init_db() -> None:
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

    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `shggzy_policy` (
                `id` BIGINT NOT NULL AUTO_INCREMENT,
                `channel` VARCHAR(32) NOT NULL DEFAULT '{POLICY_CHANNEL}' COMMENT '栏目：gjzcfg 国家政策法规等',
                `title` VARCHAR(500) NOT NULL COMMENT '标题',
                `publish_date` DATE DEFAULT NULL COMMENT '发布日期',
                `publish_time` DATETIME DEFAULT NULL COMMENT '发布时间（若页面提供）',
                `detail_url` VARCHAR(1000) NOT NULL COMMENT '详情链接',
                `url_hash` CHAR(32) NOT NULL COMMENT 'URL MD5 去重',
                `source_label` VARCHAR(300) DEFAULT NULL COMMENT '信息来源',
                `body_text` LONGTEXT COMMENT '正文纯文本',
                `body_html` LONGTEXT COMMENT '正文 HTML',
                `crawl_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '抓取时间',
                PRIMARY KEY (`id`),
                UNIQUE KEY `uk_channel_url_hash` (`channel`, `url_hash`),
                KEY `idx_publish_date` (`publish_date`),
                KEY `idx_crawl_time` (`crawl_time`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='上海公共资源交易平台-政策法规';
            """
        )
    conn.commit()
    conn.close()
    logger.info("数据库表 shggzy_policy 就绪")


def list_page_url(page_no: int) -> str:
    if page_no <= 1:
        return POLICY_LIST_FIRST
    return f"{BASE_URL}/gjzcfg_{page_no}.jhtml"


def parse_policy_list(html: str) -> tuple[list[dict[str, Any]], bool]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []
    rows = soup.select(".gui-title-bottom ul li") or soup.select(".cs-table .gui-title-bottom ul li")
    for row in rows:
        onclick = row.get("onclick", "")
        m = re.search(r"window\.open\('([^']+)'\)", onclick)
        if not m:
            continue
        href = m.group(1).strip()
        if not href:
            continue
        title_el = row.select_one("span.cs-span2") or row.select_one("p.color3 span:last-child")
        title = title_el.get_text(" ", strip=True) if title_el else ""
        date_text = ""
        for span in row.select("span"):
            t = span.get_text(strip=True)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", t):
                date_text = t
                break
        if not title:
            continue
        if href.startswith("http"):
            detail_url = href
        elif href.startswith("/"):
            detail_url = BASE_URL + href
        else:
            detail_url = BASE_URL + "/" + href

        pub_date = None
        if date_text:
            try:
                pub_date = datetime.strptime(date_text, "%Y-%m-%d").date()
            except ValueError:
                pass

        records.append(
            {
                "title": title,
                "publish_date": pub_date,
                "list_date_text": date_text,
                "detail_url": detail_url,
            }
        )

    m_count = re.search(r"var\s+showCount\s*=\s*(\d+)", html)
    m_limit = re.search(r"var\s+limit\s*=\s*(\d+)", html)
    m_curr = re.search(r",curr:\s*(\d+)", html)
    if m_count and m_limit and m_curr:
        total_count = int(m_count.group(1))
        page_size = int(m_limit.group(1))
        curr_page = int(m_curr.group(1))
        has_next = curr_page * page_size < total_count
    else:
        has_next = bool(records)

    return records, has_next


def _parse_detail_meta(title_p_text: str) -> tuple[Any, Any, str | None]:
    """从 content-box 下 title_p 解析发布时间与信息来源"""
    publish_time = None
    publish_date = None
    source_label = None
    compact = re.sub(r"\s+", " ", title_p_text.replace("\xa0", " ").replace("\u3000", " "))
    m_t = re.search(r"发布时间[：:]\s*([\d\-:\s]{8,25})", compact)
    if m_t:
        raw = m_t.group(1).strip()
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                publish_time = datetime.strptime(raw[:19].strip(), fmt)
                publish_date = publish_time.date()
                break
            except ValueError:
                continue
    m_s = re.search(r"信息来源[：:]\s*(.+?)(?:\s+浏览次数|$)", compact)
    if m_s:
        source_label = re.sub(r"\s+", " ", m_s.group(1).strip()).strip() or None
    return publish_time, publish_date, source_label


def parse_policy_detail(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, Any] = {
        "title": "",
        "publish_time": None,
        "publish_date": None,
        "source_label": None,
        "body_text": "",
        "body_html": "",
    }
    box = soup.select_one(".content-box")
    if not box:
        return out

    h2 = box.select_one("h2")
    if h2:
        out["title"] = h2.get_text(" ", strip=True)

    title_p = box.select_one("p.title_p")
    if title_p:
        pt, pd, src = _parse_detail_meta(title_p.get_text(" ", strip=True))
        out["publish_time"] = pt
        out["publish_date"] = pd
        out["source_label"] = src

    content = box.select_one("div.content")
    if content:
        out["body_html"] = str(content)
        out["body_text"] = content.get_text("\n", strip=True)

    return out


def fetch_list_page(page_no: int, session: requests.Session) -> str | None:
    url = list_page_url(page_no)
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = session.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            if not resp.text.strip():
                raise requests.RequestException("empty list body")
            return resp.text
        except requests.RequestException as e:
            if attempt >= REQUEST_RETRIES:
                logger.warning(f"列表页失败 page={page_no} url={url} err={e}")
                return None
            time.sleep(1.2 * attempt)
    return None


def fetch_detail(url: str, session: requests.Session) -> dict[str, Any]:
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = session.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            if not resp.text.strip():
                raise requests.RequestException("empty detail body")
            return parse_policy_detail(resp.text)
        except requests.RequestException as e:
            if attempt >= REQUEST_RETRIES:
                logger.warning(f"详情页失败 url={url} err={e}")
                return {}
            time.sleep(1.2 * attempt)
    return {}


def save_policy_rows(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """rows: 每条含 list 字段 + detail 解析字段 + detail_url, url_hash。返回 (inserted, updated) 粗略计数。"""
    if not rows:
        return 0, 0
    conn = _get_conn()
    inserted = updated = 0
    sql = """
        INSERT INTO `shggzy_policy`
            (channel, title, publish_date, publish_time, detail_url, url_hash,
             source_label, body_text, body_html)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            publish_date = COALESCE(VALUES(publish_date), publish_date),
            publish_time = COALESCE(VALUES(publish_time), publish_time),
            source_label = COALESCE(NULLIF(VALUES(source_label), ''), source_label),
            body_text = IF(VALUES(body_text) IS NOT NULL AND VALUES(body_text) != '', VALUES(body_text), body_text),
            body_html = IF(VALUES(body_html) IS NOT NULL AND VALUES(body_html) != '', VALUES(body_html), body_html)
    """
    try:
        with conn.cursor() as cur:
            for r in rows:
                try:
                    cur.execute(
                        sql,
                        (
                            POLICY_CHANNEL,
                            r["title"][:500],
                            r.get("publish_date"),
                            r.get("publish_time"),
                            r["detail_url"],
                            r["url_hash"],
                            r.get("source_label"),
                            r.get("body_text") or "",
                            r.get("body_html") or "",
                        ),
                    )
                    # MySQL: 1=新插入, 2=更新已存在行
                    if cur.rowcount == 1:
                        inserted += 1
                    elif cur.rowcount == 2:
                        updated += 1
                except Exception as e:
                    logger.warning(f"单条入库失败: {r.get('detail_url')} {e}")
        conn.commit()
    finally:
        conn.close()
    return inserted, updated


def merge_record(list_rec: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    title = detail.get("title") or list_rec.get("title") or ""
    pub_date = detail.get("publish_date") or list_rec.get("publish_date")
    merged = {
        "title": title,
        "publish_date": pub_date,
        "publish_time": detail.get("publish_time"),
        "source_label": detail.get("source_label"),
        "body_text": detail.get("body_text") or "",
        "body_html": detail.get("body_html") or "",
        "detail_url": list_rec["detail_url"],
        "url_hash": _url_hash(list_rec["detail_url"]),
    }
    return merged


def crawl(*, dry_run: bool = False, max_pages: int | None = None) -> None:
    limit_pages = max_pages if max_pages is not None else MAX_PAGES
    session = requests.Session()
    total_ins = total_skip = 0

    for page_no in range(1, limit_pages + 1):
        logger.info(f"列表 第 {page_no} 页: {list_page_url(page_no)}")
        html = fetch_list_page(page_no, session)
        if not html:
            break
        records, has_next = parse_policy_list(html)
        logger.info(f"  解析 {len(records)} 条, has_next={has_next}")
        if not records:
            break

        batch: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            logger.info(f"  详情 [{i+1}/{len(records)}] {rec['title'][:40]}...")
            detail = fetch_detail(rec["detail_url"], session)
            merged = merge_record(rec, detail)
            batch.append(merged)
            _sleep()
            if dry_run and page_no == 1 and i >= 0:
                logger.info(f"  [dry-run] 首条标题={merged['title'][:80]} 正文长度={len(merged.get('body_text') or '')}")
                if i == 0:
                    break

        if dry_run:
            logger.info("dry-run 结束，未写入数据库")
            return

        ins, upd = save_policy_rows(batch)
        total_ins += ins
        total_skip += upd
        logger.info(f"  入库 新增 {ins} 条，更新 {upd} 条")

        if not has_next:
            logger.info("已到最后一页")
            break
        _sleep()

    logger.success(f"完成: 累计新增 {total_ins} 条，更新 {total_skip} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description="上海公共资源交易平台 — 国家政策法规爬虫")
    parser.add_argument("--dry-run", action="store_true", help="只抓第1页且仅1条详情，不入库")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=f"最大列表页数，默认 {MAX_PAGES}（采集到该页后停止）；传更大值可继续翻页",
    )
    args = parser.parse_args()

    logger.add(
        "shggzy_policy_crawler_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="14 days",
        encoding="utf-8",
        level="INFO",
    )
    if not args.dry_run:
        init_db()
    else:
        logger.info("dry-run：跳过数据库初始化与入库")
    crawl(dry_run=args.dry_run, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
