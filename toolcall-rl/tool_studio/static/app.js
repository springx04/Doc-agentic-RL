const $ = (id) => document.getElementById(id);
const status = $("status"), logs = $("logs"), answer = $("answer"), runButton = $("runButton"), referenceButton = $("referenceButton");
let turns = 0;
let logHistory = [];

function setStatus(text, kind) { status.textContent = text; status.className = `status ${kind}`; }
function addLog(type, title, payload) {
  const node = $("logTemplate").content.firstElementChild.cloneNode(true);
  node.classList.add(type); node.querySelector(".log-meta").textContent = title;
  const text = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  logHistory.push(`[${title}]\n${text}`);
  node.querySelector("pre").textContent = text;
  for (const path of artifactPaths(text)) {
    const link = document.createElement("a");
    link.className = "artifact-link";
    link.target = "_blank";
    link.rel = "noopener";
    link.href = `/api/artifact?path=${encodeURIComponent(path)}`;
    link.textContent = `打开产物：${path.split("\\").pop()}`;
    node.appendChild(link);
  }
  logs.appendChild(node); logs.scrollTop = logs.scrollHeight;
}
function artifactPaths(text) {
  const matches = text.match(/[A-Z]:\\[^\n"']+?\.(?:png|jpg|jpeg|webp|json|md|csv|html)/gi) || [];
  return [...new Set(matches)];
}

async function loadTools() {
  try {
    const response = await fetch("/api/tools"); const {tools, output_dir, studio_version} = await response.json();
    $("tools").replaceChildren(...tools.map((tool) => {
      const info = tool.function; const item = document.createElement("div"); item.className = "tool";
      item.innerHTML = `<code>${info.name}</code><p>${info.description}</p>`; return item;
    }));
    $("tools").title = `产物目录：${output_dir}`;
    $("version").textContent = studio_version ? `server: ${studio_version}` : "server: old version";
  } catch (error) { $("tools").textContent = `工具列表加载失败：${error.message}`; }
}

async function runAgent() {
  const prompt = $("prompt").value.trim();
  if (!prompt) { $("prompt").focus(); return; }
  const documentPath = $("documentPath").value.trim();
  const agentPrompt = documentPath ? `${prompt}\n\nDocument path: ${documentPath}` : prompt;
  const payload = { prompt: agentPrompt, max_turns: Number($("maxTurns").value), config: {
    base_url: $("baseUrl").value, api_key: $("apiKey").value, model: $("model").value,
    temperature: Number($("temperature").value), system_prompt: $("systemPrompt").value
  }};
  await runStream("/api/run", payload);
}
async function runReferenceTest() {
  const documentPath = $("documentPath").value.trim();
  if (!documentPath) { $("documentPath").focus(); return; }
  await runStream("/api/reference-test", {document_path: documentPath});
}
async function runStream(endpoint, payload) {
  logs.replaceChildren(); logHistory = []; answer.textContent = "等待运行结果…"; turns = 0; $("turnCount").textContent = "0 turns";
  runButton.disabled = true; referenceButton.disabled = true; setStatus("运行中", "running");
  try {
    const response = await fetch(endpoint, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    if (!response.ok || !response.body) throw new Error(await response.text());
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buffer += decoder.decode(value, {stream:true}); const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) if (line) handleEvent(JSON.parse(line));
    }
  } catch (error) { addLog("error", "界面请求失败", error.message); setStatus("连接失败", "error"); }
  finally { runButton.disabled = false; referenceButton.disabled = false; if (status.classList.contains("running")) setStatus("完成", "done"); }
}
function handleEvent(item) {
  if (item.turn) { turns = Math.max(turns, item.turn); $("turnCount").textContent = `${turns} turns`; }
  if (item.type === "assistant") addLog("assistant", `Turn ${item.turn} · assistant`, item.content || "(tool call without text)");
  else if (item.type === "model_progress") addLog("model_progress", `Turn ${item.turn} · model API running (${item.elapsed_seconds}s)`, item.detail);
  else if (item.type === "tool_call") addLog("tool_call", `Turn ${item.turn} · ${item.name} arguments`, item.arguments);
  else if (item.type === "tool_started") addLog("tool_started", `Turn ${item.turn} · ${item.name} started`, item.detail);
  else if (item.type === "tool_progress") addLog("tool_progress", `Turn ${item.turn} · ${item.name} running (${item.elapsed_seconds}s)`, item.detail);
  else if (item.type === "tool_result") {
    const toolStatus = item.status || resultStatus(item.result);
    const elapsed = Number.isFinite(item.elapsed_seconds) ? `${item.elapsed_seconds}s` : "duration unavailable";
    const kind = toolStatus === "error" ? "error" : "tool_result";
    addLog(kind, `Turn ${item.turn} · ${item.name} result (${elapsed}, ${toolStatus})`, item.result);
  }
  else if (item.type === "model_request") addLog("model_request", `Turn ${item.turn} · 请求模型`, {message_count:item.message_count});
  else if (item.type === "error") { addLog("error", "错误", item.message); answer.textContent = item.message; setStatus("出错", "error"); }
  else if (item.type === "completed") { answer.textContent = item.answer; setStatus("完成", "done"); addLog("completed", `完成 · ${item.turns} turns`, item.answer); }
  else if (item.type === "run_started") addLog("run_started", "Agent 已启动", {max_turns:item.max_turns, tool_count:item.tool_count});
}
function resultStatus(result) {
  try { return JSON.parse(result).status || "unknown"; } catch (_) { return "unknown"; }
}
async function copyLogs() {
  const text = logHistory.length ? logHistory.join("\n\n") : "No run logs yet.";
  try {
    await navigator.clipboard.writeText(text);
    $("copyLogsButton").textContent = "已复制";
  } catch (_) {
    const area = document.createElement("textarea"); area.value = text; document.body.appendChild(area);
    area.select(); document.execCommand("copy"); area.remove();
    $("copyLogsButton").textContent = "已复制";
  }
  setTimeout(() => { $("copyLogsButton").textContent = "复制日志"; }, 1200);
}
runButton.addEventListener("click", runAgent);
referenceButton.addEventListener("click", runReferenceTest);
$("clearButton").addEventListener("click", () => { logs.replaceChildren(); logHistory = []; answer.textContent = "尚未运行。"; turns=0; $("turnCount").textContent="0 turns"; setStatus("就绪", "idle"); });
$("copyLogsButton").addEventListener("click", copyLogs);
loadTools();
