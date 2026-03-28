const TOKEN_KEY = "qa_access_token";
const USER_KEY = "qa_username";

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function setAuth(token, username) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, username || "");
  } else {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }
}

function goLogin() {
  window.location.replace("/login");
}

if (!getToken()) {
  goLogin();
} else {
  const chatEl = document.getElementById("chat");
  const inputEl = document.getElementById("q");
  const sendBtn = document.getElementById("send");
  const clearBtn = document.getElementById("clear");
  const statusEl = document.getElementById("status");
  const agentHintEl = document.getElementById("agent-hint");
  const chatUsername = document.getElementById("chat-username");
  const btnLogout = document.getElementById("auth-logout");

  /** @type {"nl2sql"|"policy"} — 智能分类（auto）暂未开放 */
  let agentMode = "nl2sql";

  /** 多轮对话：上一轮返回的 conversation_id */
  let lastConversationId = null;

  const HINTS = {
    nl2sql:
      "库问即答：本地数据库精准查询，直接提取标准答案。",
    policy: "政策咨询：多维度资料检索萃取，智能整合生成答案。",
  };

  function updateUserLabel() {
    if (chatUsername) {
      chatUsername.textContent = localStorage.getItem(USER_KEY) || "—";
    }
  }

  async function verifySession() {
    const t = getToken();
    if (!t) {
      goLogin();
      return;
    }
    try {
      const res = await fetch("/api/v1/users/me", {
        headers: { Authorization: `Bearer ${t}` },
      });
      if (!res.ok) {
        setAuth("", "");
        goLogin();
        return;
      }
      const data = await res.json();
      if (data && data.username) localStorage.setItem(USER_KEY, data.username);
      updateUserLabel();
    } catch {
      goLogin();
    }
  }

  if (btnLogout) {
    btnLogout.addEventListener("click", () => {
      setAuth("", "");
      lastConversationId = null;
      goLogin();
    });
  }

  document.querySelectorAll(".agent-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.getAttribute("data-mode");
      if (!mode || !["nl2sql", "policy"].includes(mode)) return;
      agentMode = mode;
      document.querySelectorAll(".agent-tab").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      if (agentHintEl) agentHintEl.textContent = HINTS[agentMode];
    });
  });

  function appendMsg(role, html) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.innerHTML = html;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function setLoading(on) {
    sendBtn.disabled = on;
    inputEl.disabled = on;
    statusEl.textContent = on ? "正在思考…" : "";
    statusEl.classList.toggle("error", false);
  }

  async function send() {
    const text = (inputEl.value || "").trim();
    if (!text) return;

    const modeLabel = agentMode === "nl2sql" ? "库表查询" : "政策咨询";
    appendMsg(
      "user",
      `<div class="label">你 <span class="mode-tag">${escapeHtml(
        modeLabel
      )}</span></div><div>${escapeHtml(text)}</div>`
    );
    inputEl.value = "";
    setLoading(true);

    try {
      const payload = { message: text, agent_mode: agentMode };
      if (lastConversationId != null) payload.conversation_id = lastConversationId;

      const headers = {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getToken()}`,
      };

      const res = await fetch("/api/chat", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));

      if (res.status === 401) {
        setAuth("", "");
        goLogin();
        return;
      }

      if (!res.ok) {
        const d = data.detail;
        const msg =
          typeof d === "string"
            ? d
            : Array.isArray(d)
            ? d.map((x) => x.msg || x).join("; ")
            : res.statusText || "请求失败";
        throw new Error(msg);
      }

      if (data.conversation_id != null && data.conversation_id !== "") {
        const cid = Number(data.conversation_id);
        if (!Number.isNaN(cid)) lastConversationId = cid;
      }

      if (data.error) {
        appendMsg(
          "bot",
          `<div class="label">甄甄</div><div class="error">${escapeHtml(
            data.error
          )}</div>`
        );
        return;
      }

      const chartBlock =
        data.chart && typeof data.chart === "object"
          ? `<div class="chart-wrap" role="img" aria-label="数据图表"></div>`
          : "";

      const refs =
        Array.isArray(data.references) && data.references.length > 0
          ? data.references
          : [];
      const refsBlock =
        refs.length > 0
          ? `<div class="refs"><div class="refs-title">参考来源</div><ul class="refs-list">${refs
              .map((r) => {
                const url = String(r.url || "").trim();
                const title = String(r.title || url).trim();
                if (!url) return "";
                return `<li><a href="${safeHref(
                  url
                )}" target="_blank" rel="noopener noreferrer">${escapeHtml(
                  title
                )}</a></li>`;
              })
              .filter(Boolean)
              .join("")}</ul></div>`
          : "";

      appendMsg(
        "bot",
        `<div class="label">甄甄</div><div>${escapeHtml(
          data.answer || "(空)"
        )}</div>${chartBlock}${refsBlock}`
      );

      if (data.chart && typeof data.chart === "object" && window.echarts) {
        requestAnimationFrame(() => {
          const el = chatEl.querySelector(".msg.bot:last-child .chart-wrap");
          if (!el) return;
          const inst = echarts.init(el);
          inst.setOption({
            backgroundColor: "transparent",
            textStyle: { color: "#e6edf3" },
            ...data.chart,
          });
          if (typeof ResizeObserver !== "undefined") {
            const ro = new ResizeObserver(() => inst.resize());
            ro.observe(el);
          } else {
            window.addEventListener("resize", () => inst.resize());
          }
        });
      }
    } catch (e) {
      statusEl.textContent = String(e.message || e);
      statusEl.classList.add("error");
      appendMsg(
        "bot",
        `<div class="label">甄甄</div><div class="error">${escapeHtml(
          String(e.message || e)
        )}</div>`
      );
    } finally {
      setLoading(false);
    }
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function safeHref(u) {
    const s = String(u || "").trim();
    if (!/^https?:\/\//i.test(s)) return "#";
    return s.replace(/"/g, "%22");
  }

  sendBtn.addEventListener("click", send);
  clearBtn.addEventListener("click", () => {
    chatEl.innerHTML = "";
    lastConversationId = null;
    statusEl.textContent = "";
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  fetch("/api/health")
    .then((r) => r.json())
    .then((j) => {
      if (j.models_loaded) {
        if (j.qa_rag_probe) {
          statusEl.textContent =
            "RAG 探针模式已就绪：仅测向量检索+生成（无需数据库）";
          statusEl.classList.remove("error", "warn");
        } else if (j.qa_lite_nl2sql_only && j.sql_cursor_ok === false) {
          statusEl.textContent =
            "模型已就绪，但数据库未连接（精简模式需 NL2SQL）。请配置 MySQL 或在 data 下放 CompanyTable.csv";
          statusEl.classList.remove("error");
          statusEl.classList.add("warn");
        } else {
          statusEl.textContent = "模型已就绪，可提问";
          statusEl.classList.remove("error", "warn");
        }
      } else if (j.smoke) {
        statusEl.textContent =
          "未加载模型：--smoke 仅测页面。请去掉 --smoke，用 start_procurement_fastapi*.bat 完整启动";
        statusEl.classList.remove("error");
        statusEl.classList.add("warn");
      } else {
        statusEl.textContent =
          "未加载模型：请查看终端启动日志，或确认 7860 未被其它进程占用";
        statusEl.classList.remove("error");
        statusEl.classList.add("warn");
      }
    })
    .catch(() => {
      statusEl.textContent = "无法连接后端";
      statusEl.classList.add("error");
    });

  updateUserLabel();
  verifySession();
}
