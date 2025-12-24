# attr-synth

Synthetic user-journey generator + Markov models.

## End-to-end steps

### 1) Generate data (store-like journeys)

Creates a flattened transitions JSON (one transition object per line inside a JSON array).

```bash
python3 store_generate.py --output transitions_store.json
```

### 2) Build a higher-order Markov chain model (order‑K) from the transitions

This produces a compact JSON with `counts` for orders `0..K` (backoff-ready). The React UI consumes this file.

```bash
python3 markov_chain.py --input transitions_store.json --order 6 --output markov_order6.json
```

### 3) Use the UI (React) to explore/predict journeys

```bash
cd webapp
npm install
npm run dev
```

Then upload the generated `markov_order6.json` in the UI.

## Optional: First-order Markov chain graph (visualization)

```bash
python3 markov_chain.py \
  --input transitions_store.json \
  --output markov_chain.json \
  --graph-output markov_chain.dot \
  --min-prob 0.02 --top-k 5 --hide-self-loops \
  --include-durations --label-dwell
```

If you have Graphviz installed, you can render:

```bash
python3 markov_chain.py --input transitions_store.json --output markov_chain.json --render svg
```
