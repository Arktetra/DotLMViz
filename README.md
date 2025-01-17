# DoTLMViz

Decoder-only Transformer Language Model Visualization.

## Setup

Run `pip install -e .` to start with the project, and then

```bash
pre-commit install
pre-commit autoupdate
```

### Frontend

```bash
cd frontend
npm i
npm run dev
```

Open the web browser and go to `http://localhost:5173/`

**Prettier (Frontend)**
To write
```bash
npm run format
```
To check
```bash
npm run lint
```

<hr />

Open a new terminal, go the root directory and run the backend:

```bash
flask --app backend run --debug
```

## Docs

To serve the docs, run `mkdocs serve`.
