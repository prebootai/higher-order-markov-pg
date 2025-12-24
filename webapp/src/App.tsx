import React, { useMemo, useState } from "react";

type CountsRow = Record<string, number>;
type CountsByOrder = Record<string, Record<string, CountsRow>>;

type MarkovModel = {
  order: number;
  delimiter: string;
  root_key: string;
  states: string[];
  meta?: {
    total_sequences?: number;
    total_transitions?: number;
  };
  counts: CountsByOrder;
};

type Prediction = {
  usedOrder: number;
  contextKey: string;
  next: Array<{ state: string; prob: number; count: number }>;
};

function formatPct(p: number) {
  return `${(p * 100).toFixed(2)}%`;
}

function sumCounts(row: CountsRow): number {
  let s = 0;
  for (const k of Object.keys(row)) s += row[k] ?? 0;
  return s;
}

function predictNext(model: MarkovModel, journey: string[], topN = 12): Prediction | null {
  const k = model.order;
  const delim = model.delimiter;
  const root = model.root_key;

  for (let o = Math.min(k, journey.length); o >= 0; o--) {
    const key = o === 0 ? root : journey.slice(-o).join(delim);
    const table = model.counts[String(o)];
    const row = table?.[key];
    if (!row) continue;

    const total = sumCounts(row);
    if (total <= 0) continue;

    const items = Object.entries(row)
      .map(([state, count]) => ({ state, count, prob: count / total }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, topN);

    return { usedOrder: o, contextKey: key, next: items };
  }

  return null;
}

async function readJsonFile(file: File): Promise<unknown> {
  const text = await file.text();
  return JSON.parse(text);
}

function isModel(x: any): x is MarkovModel {
  return (
    x &&
    typeof x === "object" &&
    typeof x.order === "number" &&
    typeof x.delimiter === "string" &&
    typeof x.root_key === "string" &&
    Array.isArray(x.states) &&
    x.counts &&
    typeof x.counts === "object"
  );
}

export function App() {
  const [model, setModel] = useState<MarkovModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [journey, setJourney] = useState<string[]>(["START"]);
  const [nextPick, setNextPick] = useState<string>("");

  const isTerminal = (s: string) => s === "NULL" || s === "CONVERT";
  const journeyEnded = journey.length > 0 && isTerminal(journey[journey.length - 1]);

  const prediction = useMemo(() => {
    if (!model) return null;
    if (journeyEnded) return null;
    return predictNext(model, journey, 12);
  }, [model, journey, journeyEnded]);

  const allStates = model?.states ?? [];
  const addableStates = allStates.filter((s) => s !== "START");

  function resetJourney() {
    setJourney(["START"]);
  }

  function popState() {
    setJourney((j) => (j.length > 1 ? j.slice(0, -1) : j));
  }

  function appendState(s: string) {
    if (!s) return;
    setJourney((j) => {
      if (j.length > 0 && isTerminal(j[j.length - 1])) return j;
      if (j.length === 0) return ["START", s];
      if (j.length === 1 && j[0] !== "START") return ["START", s];
      return [...j, s];
    });
  }

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setError(null);
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const parsed = await readJsonFile(file);
      if (!isModel(parsed)) {
        throw new Error("Model JSON not recognized. Expected output from markov_chain.py.");
      }
      setModel(parsed);
      setJourney(["START"]);
      setNextPick("");
    } catch (err: any) {
      setModel(null);
      setError(err?.message ?? String(err));
    }
  }

  const modelStats = useMemo(() => {
    if (!model) return null;
    const ctxCounts: Array<{ order: number; contexts: number }> = [];
    for (let o = 0; o <= model.order; o++) {
      const table = model.counts[String(o)];
      ctxCounts.push({ order: o, contexts: table ? Object.keys(table).length : 0 });
    }
    return ctxCounts;
  }, [model]);

  return (
    <div className="container">
      <div className="topbar">
        <div>
          <div className="title">Attr Synth — Higher‑Order Markov Viewer</div>
          <div className="subtitle">
            Load an order‑K model JSON from markov_chain.py, then build a journey and view next‑action predictions (with backoff).
          </div>
        </div>
        <div className="row">
          {model ? (
            <span className="pill good">
              Model loaded (order {model.order}) • {model.states.length} states
            </span>
          ) : (
            <span className="pill warn">No model loaded</span>
          )}
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h2>Load model</h2>
          <input type="file" accept="application/json" onChange={onFileChange} />
          {error && (
            <div style={{ marginTop: 10 }} className="pill warn">
              {error}
            </div>
          )}

          {model && modelStats && (
            <div style={{ marginTop: 12 }} className="row">
              <span className="pill">
                sequences: {model.meta?.total_sequences?.toLocaleString?.() ?? "?"}
              </span>
              <span className="pill">
                transitions: {model.meta?.total_transitions?.toLocaleString?.() ?? "?"}
              </span>
              <span className="pill">delimiter: <span className="mono">{model.delimiter}</span></span>
              <span className="pill">root: <span className="mono">{model.root_key}</span></span>
            </div>
          )}
          {model && modelStats && (
            <div style={{ marginTop: 12 }}>
              <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
                Context tables loaded
              </div>
              <div className="chips">
                {modelStats.map((s) => (
                  <span key={s.order} className="chip">
                    <strong>o{s.order}</strong> {s.contexts.toLocaleString()} ctx
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Journey</h2>
          <div className="chips" style={{ marginBottom: 10 }}>
            {journey.map((s, idx) => (
              <span className="chip" key={`${s}-${idx}`}>
                {idx === 0 ? <strong>{s}</strong> : s}
              </span>
            ))}
          </div>

          <div className="row" style={{ marginBottom: 10 }}>
            <button className="btn danger" onClick={resetJourney}>
              Reset to START
            </button>
            <button className="btn" onClick={popState}>
              Remove last
            </button>
          </div>

          <div className="row" style={{ marginBottom: 10 }}>
            <select
              className="select"
              value={nextPick}
              disabled={!model || journeyEnded}
              onChange={(e) => setNextPick(e.target.value)}
            >
              <option value="">Select next state…</option>
              {addableStates.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <button
              className="btn primary"
              disabled={!model || !nextPick || journeyEnded}
              onClick={() => appendState(nextPick)}
            >
              Add
            </button>
          </div>

          {prediction && (
            <div style={{ marginTop: 8 }} className="row">
              <span className="pill">
                used order: <span className="mono">{prediction.usedOrder}</span>
              </span>
              <span className="pill">
                context: <span className="mono">{prediction.contextKey}</span>
              </span>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Next action prediction</h2>
          {!model && <div className="muted">Load a model JSON to enable predictions.</div>}
          {model && journeyEnded && (
            <div className="muted">
              Journey ended at a terminal state. Remove the last state or reset to continue.
            </div>
          )}
          {model && !journeyEnded && !prediction && (
            <div className="muted">
              No prediction found for this context (even after backoff). Try different states or rebuild the model.
            </div>
          )}

          {model && !journeyEnded && prediction && (
            <>
              <div className="predList" style={{ marginTop: 10 }}>
                {prediction.next.map((x) => (
                  <div key={x.state} className="predRow">
                    <div className="row" style={{ justifyContent: "space-between", gap: 10 }}>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 650 }}>{x.state}</div>
                        <div className="muted" style={{ fontSize: 12 }}>
                          {formatPct(x.prob)} • n={x.count}
                        </div>
                      </div>
                      <button className="btn" onClick={() => appendState(x.state)} disabled={journeyEnded}>
                        Append
                      </button>
                    </div>
                    <div className="bar" title={formatPct(x.prob)}>
                      <div style={{ width: `${Math.max(1, x.prob * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 12 }} className="muted">
                Tip: if the exact 6‑order context is rare, the app automatically backs off to lower orders.
              </div>
            </>
          )}
        </div>

        <div className="card">
          <h2>How to build the model</h2>
          <div className="muted" style={{ lineHeight: 1.5 }}>
            Build an order‑K model JSON with <span className="mono">markov_chain.py</span>, then upload it here.
          </div>
          <div className="mono" style={{ marginTop: 10, whiteSpace: "pre-wrap" }}>
            python3 markov_chain.py --input transitions_store.json --order 6 --output markov_order6.json
          </div>
          <div className="muted" style={{ marginTop: 10 }}>
            Prefer <span className="mono">transitions_store.json</span> (it has session boundaries).
          </div>
        </div>
      </div>
    </div>
  );
}


