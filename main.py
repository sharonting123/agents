# [01] 阅读顺序编号文件，对应原文件: main.py
import argparse
import os

ARGS = None


def _parse_args():
    p = argparse.ArgumentParser(description="政府采购投标助手：分类→关键词→SQL→答题（批量评测）")
    p.add_argument("--gpu", type=int, default=0, help="CUDA 设备序号（CUDA_VISIBLE_DEVICES）")
    return p.parse_args()


def main():
    global ARGS
    ARGS = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)

    import torch
    from datetime import datetime
    from loguru import logger

    from config import cfg
    from chatglm_ptuning import ChatGLM_Ptuning, PtuningType
    from generate_answer_with_classify import do_gen_keywords
    from generate_answer_with_classify import (
        do_classification,
        do_sql_generation,
        generate_answer,
        make_answer,
    )

    def check_paths():
        os.makedirs(cfg.DATA_PATH, exist_ok=True)
        if not os.path.exists(cfg.DATA_PATH):
            raise RuntimeError("DATA_PATH not exists: {}".format(cfg.DATA_PATH))

        print("Torch cuda available ", torch.cuda.is_available())

        for name, path in (("NL2SQL_CHECKPOINT_PATH", cfg.NL2SQL_CHECKPOINT_PATH),):
            if not os.path.exists(path):
                raise RuntimeError("{} not exists: {}".format(name, path))

        print("Check paths success!")

    DATE = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(cfg.DATA_PATH, "{}.main.log".format(DATE))
    os.makedirs(cfg.DATA_PATH, exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)
    logger.add(log_path, level="DEBUG")

    check_paths()

    cls_m = ChatGLM_Ptuning(PtuningType.Classify)
    do_classification(cls_m)
    cls_m.unload_model()

    class _KwStub:
        isKeywords = False

    do_gen_keywords(_KwStub())

    sql_m = ChatGLM_Ptuning(PtuningType.NL2SQL)
    do_sql_generation(sql_m)
    sql_m.unload_model()

    chat = ChatGLM_Ptuning(PtuningType.Nothing)
    generate_answer(chat)
    chat.unload_model()

    make_answer()
    logger.info("主流程结束，日志: {}", log_path)


if __name__ == "__main__":
    main()
