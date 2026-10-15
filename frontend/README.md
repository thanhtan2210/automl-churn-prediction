# Frontend (Next.js)

Four pages: Upload (`/`), Profile (`/datasets/[id]`), Model (`/datasets/[id]/model`), Predict (`/datasets/[id]/predict`).
Start the API first (`python -m uvicorn src.api.main:app --port 8000` from the repo root), then `npm ci && npm run dev`.
The API URL comes from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`); set it in `frontend/.env.local`.
Checks used in CI: `npm run lint` and `npm run build`.
