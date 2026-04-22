# HackWatch — Frontend Engineer Prompt

> Give this prompt + the contents of `demo/build/index.html` to a new Claude session.

---

## Context

You are a Senior Frontend Engineer working on **HackWatch**, a real-time AI reward-hacking detection demo built for the Meta PyTorch OpenEnv Hackathon 2026.

The app shows a **MONITOR agent** watching a **WORKER agent** solve coding tasks in real time — detecting when the worker cheats (reward hacks) by reading its code diffs.

---

## Stack Constraints — Non-Negotiable

- **Single file:** `demo/build/index.html`
- **NO JSX.** The file was pre-transpiled. All React code must use `React.createElement()` exclusively. JSX syntax will cause a parse error at runtime.
- **NO build step, NO npm, NO bundler, NO import/export statements.**
- React 18 UMD loaded from `/demo/js/react.min.js`
- ReactDOM 18 UMD loaded from `/demo/js/react-dom.min.js`
- No additional `<script>` tags or CDN links.
- All CSS additions go inside the existing `<style>` block.

---

## Design Direction — Hybrid of Two References

You are merging the best of two existing designs. Rewrite the CSS completely to achieve this hybrid. Do not keep the old CSS as-is.

**From v11 (amber CRT — keep these):**
- Dark near-black backgrounds: `#080706` page, `#0e0c09` panels
- Amber as the **brand color**: logo, header accents, active states, progress bar, cursor
- Dramatic exploit animations: pulsing red border glow on verdict card, blinking red corner triangle
- CRT phosphor scanlines overlay (subtle: `rgba(0,0,0,.15)` repeating gradient, `mix-blend-mode: multiply`)
- Diamond-shaped sparkline dots (`clip-path: polygon(50% 0%,100% 50%,50% 100%,0% 50%)`)
- Amber top border line on header (`linear-gradient` with `box-shadow: 0 0 12px`)
- Blinking block cursor in terminal

**From current.png (navy green — adopt these):**
- **Typography**: `JetBrains Mono` for all code/mono, `Inter` for body text and labels — load both from Google Fonts
- **Logo**: `[ HACKWATCH ]` with brackets dimmer than the text, `font-weight: 700`, `letter-spacing: 0.12em` — keep amber color
- **Green for clean states, red for exploit** — reward bars, verdict card, sparkline dots, right panel border all use green (`#00ff88`) for clean and red (`#ff4455`) for exploit — NOT amber for everything
- **Gradient reward bars** with matching glow: green bars `linear-gradient(90deg, #00cc66, #00ff88)`, red bars `linear-gradient(90deg, #cc2233, #ff4455)`, amber bars `linear-gradient(90deg, #b07800, #e8a000)` — each with `box-shadow` glow
- **Diff view**: green-tinted `+` lines `rgba(0,255,136,.08)` with `color: #66ffaa`, red-tinted `-` lines `rgba(255,68,85,.08)` with `color: #ff7766` — much more readable than the current flat amber
- **Rounded corners** everywhere: panels `border-radius: 6px`, verdict card `border-radius: 8px`, badges `border-radius: 4px`, run button `border-radius: 5px`
- **Donut chart**: circular legend dots (not diamond), `border-radius: 50%`
- **Sparkline**: round connector line between dots (1px, `var(--border)`) — keep diamond dot shape from v11
- **Panel headers**: uppercase `letter-spacing: 0.12em`, `font-size: 10px`, clean separator line
- **Run button**: green-tinted `background: rgba(0,255,136,.1)`, `border: 1px solid rgba(0,255,136,.35)`, `color: #00ff88`, `border-radius: 5px`

**New combined `:root` variables — replace the old ones entirely:**
```css
:root {
  --bg:          #080706;
  --bg2:         #0d0d12;
  --bg3:         #121218;
  --bg4:         #17171f;
  --border:      #1e1e2a;
  --border2:     #2a2a3a;
  --amber:       #e8a000;
  --amber2:      #ffcc44;
  --amber-dim:   rgba(232,160,0,.12);
  --green:       #00ff88;
  --green-dim:   rgba(0,255,136,.12);
  --red:         #ff4455;
  --red-dim:     rgba(255,68,85,.12);
  --blue:        #4488ff;
  --text:        #e8e8f0;
  --text-dim:    #8888aa;
  --text-muted:  #44445a;
  --mono:        'JetBrains Mono', monospace;
  --sans:        'Inter', sans-serif;
}
```

**Font loading (replace existing Google Fonts link):**
```html
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
```

---

## Backend API Contract

```
POST /reset    body: {} | {"seed": number}
  → MonitorObservation {
      task_prompt: string,
      worker_last_action: { tool: string, path?: string },
      worker_diff: string,       // raw unified diff — UNTRUSTED STRING
      tools_used_so_far: string[],
      step_idx: number,
      episode_done: boolean
    }

POST /step     body: { verdict, confidence, reasoning }
  → {
      observation: MonitorObservation,
      reward: number,            // 0.0–1.0
      done: boolean,
      info: { planted_label?: string }
    }
```

**Security rule:** `worker_diff` and `task_prompt` are agent-generated strings.
Treat them as **untrusted input**. Never use `innerHTML` or `dangerouslySetInnerHTML`.
Render all diff/task content as text nodes only.

---

## What Already Works — Preserve the Logic, Rewrite the Styles

Keep all existing React state and API logic unchanged. Rewrite the CSS classes and inline styles to match the new hybrid design.

- `handleRun()` calls `/reset` then one `/step`, shows real diff + reward
- Left panel: live episode log (`epLog` state)
- Right panel: `VerdictCard`, `Sparkline`, `ExploitDonut`, session stats
- Progress bar animates during fetch
- `DiffView` animates diff rows line by line
- `TrainingCurve` in footer — illustrative only, leave untouched

---

## Features to Implement

### 1 — Multi-Step Episode Loop
After `/reset`, loop `POST /step` until `response.done === true`.

- Each iteration: update `obs` state, re-render `DiffView` with the new diff, append to `epLog`
- Add `cancelRef = useRef(false)` — clicking the run button while running cancels the loop; button shows `[ CANCEL ]` during run
- Show `STEP N` counter in the center header
- Hard cap: 20 steps max, then break and log a warning
- Guard: validate `obs.worker_diff` is a non-empty string before passing to diff parser

### 2 — Confidence Slider
State: `confidence`, default `0.75`. Sent in every `/step` body.

- Placement: right panel, between `VerdictCard` and verdict buttons
- `<input type="range" min="0" max="1" step="0.01">`
- Label: `CONF  0.75` (live, `font-family: var(--mono)` 10px, `color: var(--text-dim)`)
- CSS `appearance: none`: amber thumb `var(--amber2)`, dark track `var(--border2)`, height 4px

### 3 — Verdict Selector
State: `verdict`, default `"flag"`. Buttons: `ALLOW` · `FLAG` · `BLOCK`

- Active amber (ALLOW/FLAG): `border: 1px solid rgba(232,160,0,.5)`, `background: rgba(232,160,0,.1)`, `color: var(--amber2)`, `border-radius: 4px`
- Active red (BLOCK): `border: 1px solid rgba(255,68,85,.5)`, `background: rgba(255,68,85,.1)`, `color: var(--red)`, `border-radius: 4px`
- Inactive: `border: 1px solid var(--border2)`, `color: var(--text-muted)`
- Font: `var(--mono)` 10px, `letter-spacing: .08em`
- BLOCK stops the step loop immediately after sending

### 4 — Episode History (left panel tab)
Two tabs: `LOG` and `HISTORY` at top of left panel.

- Active tab: `border-bottom: 2px solid var(--amber)`, `color: var(--amber)`, `font-family: var(--mono)` 9px
- LOG → existing `epLog` rows
- HISTORY → last 10 completed episodes

Each history row: ep number · first 38 chars of task_prompt · badge · reward
- `[EXPLOIT]` badge: `color: var(--red)`, `background: var(--red-dim)`, `border-radius: 3px`
- `[CLEAN]` badge: `color: var(--green)`, `background: var(--green-dim)`, `border-radius: 3px`
- Click re-runs `/reset` with stored seed (`seed = Math.random() * 99999 | 0`)
- Selected row: `background: rgba(232,160,0,.07)`, `border: 1px solid rgba(232,160,0,.3)`, `border-radius: 5px`

### 5 — Reasoning Textarea
State: `reasoning`, default `""`. Sent as `reasoning` in every `/step` body.

- Below verdict buttons, above separator, in right panel
- `<textarea rows={3} maxLength={280}>`, placeholder: `describe anomaly...`
- `background: var(--bg3)`, `border: 1px solid var(--border2)`, `color: var(--text-dim)`, `font-family: var(--mono)` 10px, `resize: none`, `border-radius: 4px`, `padding: 6px 8px`
- Focus: `border-color: rgba(232,160,0,.4)`, `box-shadow: 0 0 8px rgba(232,160,0,.15)`, `outline: none`

### 6 — Exploit Detection Banner
Trigger: any `/step` response where `reward < 0.5`.

- Fixed position at top of center panel, between header and body
- Content: `⚠  REWARD HACK DETECTED`
- `background: rgba(255,68,85,.1)`, `border-bottom: 1px solid rgba(255,68,85,.4)`, `color: var(--red)`, `font-family: var(--mono)` 12px, `letter-spacing: .12em`, `text-shadow: 0 0 14px rgba(255,68,85,.6)`
- Fade in 150ms, auto-dismiss after 3s with 400ms fade out
- Reset timer if a new exploit fires while banner is visible

---

## Error Handling

- Every `fetch` in `try/catch`
- On failure: set error state, log in red, stop loop, reset progress to `0`
- Use `finally` — never leave `running = true` after error or loop exit

---

## Output Instruction

Return the **complete updated `demo/build/index.html`** file.
Do not truncate or omit any part. The file must be copy-pasteable and work immediately.
After writing the JSX, pre-transpile it: extract the `<script type="text/babel">` block, compile with Babel `@babel/preset-react`, replace with a plain `<script>` block, and remove the Babel loader `<script>` tag.
