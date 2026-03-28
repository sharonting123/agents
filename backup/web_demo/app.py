import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


MODEL_DIR = r"D:\Machinelearning\Deeplearning\0104class\D10\Code\Code\data\pretrained_models\Qwen2.5-7B-Instruct-GPTQ-Int4"


def build_model():
    """
    本地加载 Qwen2.5-7B-Instruct-GPTQ-Int4 量化模型。

    要求：
    - MODEL_DIR 指向包含 config.json / 模型权重 / tokenizer 的目录
    - 已安装 transformers，且版本支持 Qwen2.5 系列
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


tokenizer, model = build_model()


def chat_once(question: str, history: list[list[str]] | None = None):
    """
    简单的多轮对话封装。

    - history 结构：[[user, assistant], ...]
    - 目前 QwenLoRA 只接受单轮 prompt，这里把历史拼在一起作为上文。
    """
    history = history or []

    # 构造对话消息，使用 Qwen chat 模板
    messages = []
    for turn_user, turn_bot in history:
        messages.append({"role": "user", "content": turn_user})
        messages.append({"role": "assistant", "content": turn_bot})

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0, inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    history.append([question, answer])
    return history, history


def build_ui():
    with gr.Blocks(title="金融年报解读 - Qwen2.5 GPTQ 前端") as demo:
        gr.Markdown(
            """
            # 📊 金融年报解读智能交互系统（Qwen2.5-GPTQ 本地前端）

            本页面在本地直接加载 **Qwen2.5-7B-Instruct-GPTQ-Int4** 量化模型进行问答，
            适合 12GB 显存环境下的轻量体验。

            - 左侧输入你的问题，例如：`2019 年负债合计最高的公司是哪家？`
            - 模型会结合训练好的金融领域能力生成回答。
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        label="输入你的问题",
                        placeholder="例如：2019 年负债合计最高的上市公司是哪家？",
                        lines=2,
                    )
                    send_btn = gr.Button("发送", variant="primary")

                clear_btn = gr.Button("清空对话")

            with gr.Column(scale=1):
                gr.Markdown(
                    f"""
                    ### 使用说明

                    1. **环境要求**
                       - 约 12GB 显存的 NVIDIA 显卡
                       - `MODEL_DIR` 指向本地 Qwen2.5-7B-Instruct-GPTQ-Int4 目录：
                         `"{MODEL_DIR}"`
                    2. **启动方式**
                       - 激活虚拟环境：`conda activate finreport310`
                       - 进入目录：`cd D:\\Machinelearning\\Deeplearning\\0104class\\D10\\Code\\Code`
                       - 运行：`python web_demo/app.py`
                    3. **使用方式**
                       - 在左侧输入你的问题（如金融年报解读、指标分析等），点击“发送”查看回答。
                    """
                )

        state = gr.State([])  # 保存对话历史

        send_btn.click(
            fn=chat_once,
            inputs=[user_input, state],
            outputs=[chatbot, state],
        )

        user_input.submit(
            fn=chat_once,
            inputs=[user_input, state],
            outputs=[chatbot, state],
        )

        clear_btn.click(
            fn=lambda: ([], []),
            inputs=[],
            outputs=[chatbot, state],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7865, share=False)

