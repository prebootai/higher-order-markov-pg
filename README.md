# attr-synth

Synthetic user-journey generator + Markov models.

## Generate store-like journeys

Creates a flattened transitions JSON (one transition object per line inside a JSON array).

```bash
python3 store_generate.py --output transitions_store.json
```

## Build a first-order Markov chain (and a graph)

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

## Build a 6-order Markov model (for prediction with backoff)

```bash
python3 markov_chain_k.py --input transitions_store.json --order 6 --output markov_order6.json
```

## Web app (React) — journey builder + next-action prediction

```bash
cd webapp
npm install
npm run dev
```

Then upload the generated `markov_order6.json` in the UI.



