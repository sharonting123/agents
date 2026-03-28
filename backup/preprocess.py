# [06] 原项目为年报 PDF 抽取；政府采购投标助手不再包含 PDF/XPDF 流水线。
# 数据请通过 MySQL（shggzy_bid_result）与爬虫入库，或向 data/knowledge_base 放置 .txt/.md 供 F 类检索。
from loguru import logger


def main():
    logger.info(
        "06-preprocess: 已跳过。政府采购投标助手无年报 PDF 处理；请配置 MySQL 与知识库目录。"
    )


if __name__ == "__main__":
    main()
