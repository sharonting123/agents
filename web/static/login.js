const TOKEN_KEY = "qa_access_token";
const USER_KEY = "qa_username";

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function parseApiError(data, res) {
  const d = data && data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d.length) {
    return d
      .map((x) => {
        if (typeof x === "string") return x;
        const loc = Array.isArray(x.loc)
          ? x.loc.filter((p) => p !== "body").join(".")
          : x.loc
            ? String(x.loc)
            : "";
        const msg =
          x.msg ||
          x.message ||
          (typeof x.type === "string" ? x.type : "") ||
          "";
        if (loc && msg) return `${loc}: ${msg}`;
        if (msg) return msg;
        try {
          return JSON.stringify(x);
        } catch {
          return String(x);
        }
      })
      .join("；");
  }
  if (d && typeof d === "object" && d.msg) return String(d.msg);
  return res.statusText || "请求失败";
}

const authMsg = document.getElementById("auth-msg");
const formLogin = document.getElementById("form-login");
const formRegister = document.getElementById("form-register");
const btnShowLogin = document.getElementById("auth-show-login");
const btnShowRegister = document.getElementById("auth-show-register");

function showMsg(text, isError) {
  if (!authMsg) return;
  authMsg.textContent = text || "";
  authMsg.classList.toggle("error", Boolean(isError));
}

function switchPanel(register) {
  if (!formLogin || !formRegister || !btnShowLogin || !btnShowRegister) return;
  formLogin.style.display = register ? "none" : "flex";
  formRegister.style.display = register ? "flex" : "none";
  btnShowLogin.classList.toggle("active", !register);
  btnShowRegister.classList.toggle("active", register);
  showMsg("");
}

async function tryRedirectIfLoggedIn() {
  const t = getToken();
  if (!t) return;
  try {
    const res = await fetch("/api/v1/users/me", {
      headers: { Authorization: `Bearer ${t}` },
    });
    if (res.ok) {
      window.location.replace("/");
    }
  } catch {
    /* ignore */
  }
}

tryRedirectIfLoggedIn();

if (btnShowLogin) btnShowLogin.addEventListener("click", () => switchPanel(false));
if (btnShowRegister) btnShowRegister.addEventListener("click", () => switchPanel(true));

if (formLogin) {
  formLogin.addEventListener("submit", async (e) => {
    e.preventDefault();
    const u = (document.getElementById("login-username")?.value || "").trim();
    const p = document.getElementById("login-password")?.value || "";
    if (!u || !p) {
      showMsg("请输入用户名和密码", true);
      return;
    }
    showMsg("登录中…");
    try {
      const res = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: u, password: p }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        showMsg(parseApiError(data, res), true);
        return;
      }
      if (!data.access_token) {
        showMsg("登录响应异常", true);
        return;
      }
      localStorage.setItem(TOKEN_KEY, data.access_token);
      localStorage.setItem(USER_KEY, u);
      window.location.replace("/");
    } catch (err) {
      showMsg(String(err.message || err), true);
    }
  });
}

if (formRegister) {
  formRegister.addEventListener("submit", async (e) => {
    e.preventDefault();
    const u = (document.getElementById("reg-username")?.value || "").trim();
    const p = document.getElementById("reg-password")?.value || "";
    const p2 = document.getElementById("reg-password2")?.value || "";
    if (u.length < 2) {
      showMsg("用户名至少 2 个字符", true);
      return;
    }
    if (p.length < 6 || p.length > 12) {
      showMsg("密码须为 6～12 位", true);
      return;
    }
    if (p !== p2) {
      showMsg("两次输入的密码不一致", true);
      return;
    }
    showMsg("注册中…");
    try {
      const res = await fetch("/api/v1/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: u, password: p }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        showMsg(parseApiError(data, res), true);
        return;
      }
      showMsg("注册成功，请登录", false);
      switchPanel(false);
      const lu = document.getElementById("login-username");
      const lp = document.getElementById("login-password");
      if (lu) lu.value = u;
      if (lp) lp.value = "";
      formRegister.reset();
    } catch (err) {
      showMsg(String(err.message || err), true);
    }
  });
}
