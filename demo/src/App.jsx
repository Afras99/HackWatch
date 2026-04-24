const { useState, useEffect, useCallback, useRef } = React;

const API = "";  // same origin

const HACK_SIGNALS = [
  [/assert\s+True/, "test_override"],
  [/import subprocess/, "subprocess_escape"],
  [/pytest_runtest_makereport/, "conftest_tamper"],
  [/monkeypatch\.setattr.*time/, "timer_bypass"],
  [/os\.environ\[/, "env_var_injection"],
  [/_PRECOMP\s*=/, "gradient_hardcode"],
  [/eval\(/, "eval_injection"],
  [/result_cache\.json/, "file_leak"],
];

function inferHackType(diff) {
  if (!diff) return null;
  for (const [re, label] of HACK_SIGNALS) {
    if (re.test(diff)) return label;
  }
  return null;
}

function Badge({ children, color }) {
  const colors = {
    green: "background:#22c55e;color:#fff",
    red: "background:#ef4444;color:#fff",
    yellow: "background:#eab308;color:#000",
    blue: "background:#3b82f6;color:#fff",
    gray: "background:#6b7280;color:#fff",
  };
  return (
    <span style={{
      ...Object.fromEntries(colors[color].split(";").map(p => p.split(":"))),
      padding: "2px 8px", borderRadius: 9999, fontSize: 12, fontWeight: 700,
      display: "inline-block",
    }}>{children}</span>
  );
}

function VerdictButton({ value, selected, onClick }) {
  const palette = {
    allow: { bg: selected ? "#22c55e" : "#1f2937", border: "#22c55e", text: selected ? "#fff" : "#22c55e" },
    flag:  { bg: selected ? "#eab308" : "#1f2937", border: "#eab308", text: selected ? "#000" : "#eab308" },
    block: { bg: selected ? "#ef4444" : "#1f2937", border: "#ef4444", text: selected ? "#fff" : "#ef4444" },
  };
  const p = palette[value];
  return (
    <button onClick={onClick} style={{
      background: p.bg, border: `2px solid ${p.border}`, color: p.text,
      padding: "8px 20px", borderRadius: 8, cursor: "pointer",
      fontWeight: 700, fontSize: 15, transition: "all 0.15s",
    }}>{value.toUpperCase()}</button>
  );
}

function DiffView({ diff }) {
  if (!diff) return <pre style={{ color: "#6b7280", fontStyle: "italic" }}>No diff yet.</pre>;
  const lines = diff.split("\n");
  return (
    <pre style={{ margin: 0, fontSize: 13, lineHeight: 1.5, overflowX: "auto" }}>
      {lines.map((line, i) => {
        const color = line.startsWith("+") ? "#86efac" : line.startsWith("-") ? "#fca5a5" : "#9ca3af";
        return <div key={i} style={{ color }}>{line || " "}</div>;
      })}
    </pre>
  );
}

function ResultBanner({ info, diff }) {
  if (!info) return null;
  const { tp, fp, tn, fn, reward, hack_present } = info;
  const hackType = inferHackType(diff);

  let label, color, icon;
  if (tp) { label = "HACK CAUGHT"; color = "#22c55e"; icon = "🎯"; }
  else if (tn) { label = "CLEAN — CORRECTLY ALLOWED"; color = "#3b82f6"; icon = "✅"; }
  else if (fp) { label = "FALSE ALARM"; color = "#eab308"; icon = "⚠️"; }
  else if (fn) { label = "HACK MISSED"; color = "#ef4444"; icon = "❌"; }
  else return null;

  return (
    <div style={{
      border: `2px solid ${color}`, borderRadius: 10, padding: "12px 16px",
      marginTop: 12, background: color + "18",
    }}>
      <div style={{ fontSize: 18, fontWeight: 800, color }}>{icon} {label}</div>
      {hackType && (fn || tp) && (
        <div style={{ color: "#e5e7eb", fontSize: 13, marginTop: 4 }}>
          Exploit primitive: <code style={{ color: "#fbbf24" }}>{hackType}</code>
        </div>
      )}
      <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 13, color: "#9ca3af" }}>
        <span>Reward: <b style={{ color: "#fff" }}>{(reward ?? 0).toFixed(3)}</b></span>
        <span>TP:{tp} FP:{fp} TN:{tn} FN:{fn}</span>
        {info.calibration_bonus !== undefined && (
          <span>Brier bonus: <b style={{ color: "#a78bfa" }}>{(info.calibration_bonus ?? 0).toFixed(3)}</b></span>
        )}
      </div>
    </div>
  );
}

function HistoryRow({ rec }) {
  const outcome = rec.tp ? "TP" : rec.fp ? "FP" : rec.tn ? "TN" : "FN";
  const outColor = rec.tp || rec.tn ? "#22c55e" : rec.fp ? "#eab308" : "#ef4444";
  const verdictColor = rec.verdict === "allow" ? "#22c55e" : rec.verdict === "flag" ? "#eab308" : "#ef4444";
  return (
    <tr>
      <td style={{ color: "#9ca3af", fontSize: 12, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{rec.task_prompt?.slice(0, 50) || "—"}</td>
      <td style={{ color: verdictColor, fontWeight: 700, fontSize: 13 }}>{rec.verdict}</td>
      <td style={{ color: outColor, fontWeight: 700, fontSize: 13 }}>{outcome}</td>
      <td style={{ color: "#fff", fontSize: 13 }}>{(rec.reward ?? 0).toFixed(3)}</td>
      <td style={{ color: "#a78bfa", fontSize: 13 }}>{rec.conf?.toFixed(2)}</td>
    </tr>
  );
}

function App() {
  const [phase, setPhase] = useState("idle");   // idle | running | done
  const [obs, setObs] = useState(null);
  const [verdict, setVerdict] = useState("allow");
  const [confidence, setConfidence] = useState(0.7);
  const [reasoning, setReasoning] = useState("");
  const [lastResult, setLastResult] = useState(null);  // {info, reward, done}
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stepCount, setStepCount] = useState(0);

  const startEpisode = useCallback(async () => {
    setLoading(true); setError(null); setLastResult(null);
    setReasoning(""); setVerdict("allow"); setConfidence(0.7); setStepCount(0);
    try {
      const r = await fetch(`${API}/reset`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      if (!r.ok) throw new Error(`/reset ${r.status}`);
      const data = await r.json();
      setObs(data);
      setPhase("running");
    } catch (e) { setError(e.message); }
    setLoading(false);
  }, []);

  const submitVerdict = useCallback(async () => {
    if (!obs || phase !== "running") return;
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict, confidence, reasoning }),
      });
      if (!r.ok) throw new Error(`/step ${r.status}`);
      const data = await r.json();
      setStepCount(c => c + 1);
      if (data.done) {
        setLastResult(data.info);
        setPhase("done");
        setHistory(h => [{
          task_prompt: obs?.task_prompt,
          verdict, conf: confidence,
          reward: data.info?.reward,
          tp: data.info?.tp, fp: data.info?.fp,
          tn: data.info?.tn, fn: data.info?.fn,
        }, ...h.slice(0, 19)]);
      } else {
        setObs(data.observation);
        setVerdict("allow"); setConfidence(0.7); setReasoning("");
      }
    } catch (e) { setError(e.message); }
    setLoading(false);
  }, [obs, phase, verdict, confidence, reasoning]);

  const correctCount = history.filter(r => r.tp || r.tn).length;
  const acc = history.length ? (correctCount / history.length * 100).toFixed(0) : "—";

  return (
    <div style={{ minHeight: "100vh", background: "#0f172a", color: "#e5e7eb", fontFamily: "'Courier New', monospace", padding: 24 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 26, fontWeight: 900, color: "#f1f5f9" }}>
            <span style={{ color: "#ef4444" }}>Hack</span>Watch
            <span style={{ fontSize: 13, fontWeight: 400, color: "#6b7280", marginLeft: 12 }}>Monitor Demo</span>
          </h1>
          <div style={{ fontSize: 12, color: "#4b5563", marginTop: 2 }}>
            OpenEnv RL environment — reward-hacking detection · Qwen2.5-3B LoRA GRPO
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {history.length > 0 && (
            <div style={{ fontSize: 13, color: "#9ca3af" }}>
              Accuracy: <b style={{ color: acc >= 80 ? "#22c55e" : "#eab308" }}>{acc}%</b>
              <span style={{ marginLeft: 8, color: "#4b5563" }}>({history.length} eps)</span>
            </div>
          )}
          <button onClick={startEpisode} disabled={loading}
            style={{ background: "#3b82f6", color: "#fff", border: "none", padding: "10px 20px",
              borderRadius: 8, cursor: loading ? "not-allowed" : "pointer", fontWeight: 700, fontSize: 14 }}>
            {loading ? "..." : phase === "idle" ? "▶ Start Episode" : "↺ New Episode"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ background: "#450a0a", border: "1px solid #ef4444", borderRadius: 8, padding: "10px 16px", marginBottom: 16, color: "#fca5a5", fontSize: 13 }}>
          ⚠ {error} — is the server running? <code>uvicorn server.app:app --port 8000</code>
        </div>
      )}

      {phase === "idle" && !error && (
        <div style={{ textAlign: "center", padding: "60px 0", color: "#4b5563" }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🔍</div>
          <div style={{ fontSize: 18, color: "#6b7280" }}>Click <b style={{ color: "#3b82f6" }}>Start Episode</b> to begin</div>
          <div style={{ fontSize: 13, marginTop: 8, color: "#374151" }}>The monitor reviews code diffs and decides: allow / flag / block</div>
        </div>
      )}

      {(phase === "running" || phase === "done") && obs && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 20 }}>
          {/* Left panel */}
          <div>
            <div style={{ background: "#1e293b", borderRadius: 10, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#4b5563", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Task Prompt</div>
              <div style={{ fontSize: 14, color: "#e2e8f0", lineHeight: 1.6 }}>{obs.task_prompt}</div>
              <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
                <Badge color="gray">Step {(obs.step_idx ?? stepCount) + 1}</Badge>
                {obs.tools_used_so_far?.length > 0 && (
                  <span style={{ fontSize: 12, color: "#6b7280" }}>Tools: {obs.tools_used_so_far.join(", ")}</span>
                )}
              </div>
            </div>

            <div style={{ background: "#1e293b", borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 11, color: "#4b5563", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Worker Diff</div>
              <div style={{ background: "#0f172a", borderRadius: 6, padding: 12, maxHeight: 320, overflowY: "auto" }}>
                <DiffView diff={obs.worker_diff} />
              </div>
            </div>

            {phase === "done" && <ResultBanner info={lastResult} diff={obs.worker_diff} />}
          </div>

          {/* Right panel — verdict control */}
          <div>
            <div style={{ background: "#1e293b", borderRadius: 10, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#4b5563", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>Verdict</div>
              <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
                {["allow", "flag", "block"].map(v => (
                  <VerdictButton key={v} value={v} selected={verdict === v}
                    onClick={() => phase === "running" && setVerdict(v)} />
                ))}
              </div>

              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 6 }}>
                  Confidence: <b style={{ color: "#fff" }}>{confidence.toFixed(2)}</b>
                </div>
                <input type="range" min={0} max={1} step={0.01} value={confidence}
                  disabled={phase === "done"}
                  onChange={e => setConfidence(parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: "#3b82f6" }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#4b5563" }}>
                  <span>uncertain</span><span>certain</span>
                </div>
              </div>

              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 6 }}>Reasoning</div>
                <textarea value={reasoning} disabled={phase === "done"}
                  onChange={e => setReasoning(e.target.value)}
                  placeholder="What pattern did you notice?"
                  rows={4}
                  style={{ width: "100%", background: "#0f172a", border: "1px solid #374151", borderRadius: 6,
                    color: "#e5e7eb", padding: 8, fontSize: 13, fontFamily: "inherit", resize: "vertical", boxSizing: "border-box" }} />
              </div>

              <button onClick={submitVerdict} disabled={loading || phase === "done"}
                style={{ width: "100%", background: phase === "done" ? "#374151" : "#7c3aed",
                  color: "#fff", border: "none", padding: "11px 0", borderRadius: 8,
                  cursor: phase === "done" ? "default" : "pointer", fontWeight: 700, fontSize: 15 }}>
                {loading ? "Evaluating..." : phase === "done" ? "Episode Complete" : "Submit Verdict →"}
              </button>
            </div>

            {/* Reward breakdown card */}
            {lastResult && (
              <div style={{ background: "#1e293b", borderRadius: 10, padding: 16 }}>
                <div style={{ fontSize: 11, color: "#4b5563", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Reward Breakdown</div>
                {[
                  ["Detection", (lastResult.detection ?? 0) * 0.85, "#22c55e"],
                  ["Brier Bonus", lastResult.calibration_bonus ?? 0, "#a78bfa"],
                  ["Latency Bonus", lastResult.latency_bonus ?? 0, "#3b82f6"],
                  ["Cal Penalty", -(lastResult.calibration_penalty ?? 0), "#f97316"],
                  ["Intervention Cost", -(lastResult.intervention_cost ?? 0), "#ef4444"],
                  ["Total", lastResult.reward ?? 0, "#fff"],
                ].map(([label, val, color]) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4, borderTop: label === "Total" ? "1px solid #374151" : "none", paddingTop: label === "Total" ? 4 : 0 }}>
                    <span style={{ color: "#6b7280" }}>{label}</span>
                    <span style={{ color, fontWeight: label === "Total" ? 800 : 400 }}>{val >= 0 ? "+" : ""}{val.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* History table */}
      {history.length > 0 && (
        <div style={{ marginTop: 24, background: "#1e293b", borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 11, color: "#4b5563", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>
            Episode History ({history.length})
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #374151" }}>
                {["Task", "Verdict", "Outcome", "Reward", "Conf"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "4px 8px", fontSize: 11, color: "#4b5563", textTransform: "uppercase", letterSpacing: 1 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((rec, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #1e2a3a" }}>
                  <HistoryRow rec={rec} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
