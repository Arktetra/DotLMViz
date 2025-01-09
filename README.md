# DoTLMViz

Decoder-only Transformer Language Model Visualization.

## Setup

Run `pip install -e .` to start with the project, and then

```bash
pre-commit install
pre-commit autoupdate
```

## Run

Run the frontend:

```bash
cd frontend
npm run dev
```

Open the web browser and go to `http://localhost:5173/`

Open a new terminal, go the root directory and run the backend:

```bash
flask --app backend run --debug
```

## Docs

To serve the docs, run `mkdocs serve`.
