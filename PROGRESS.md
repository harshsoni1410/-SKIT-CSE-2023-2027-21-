# LipSense — Progress Log

Updated weekly. Newest entry on top.

---

## Week of 2026-09-03 — Foundation + first tasks

**Done:**
- Inspected the repo — no existing code, starting from scratch.
- Created folder structure: `team_ai_model/`, `team_video_processing/`, `team_ui/`, `demo/`, `docs/`.
- Python 3.10 virtual environment created (`venv/`), all dependencies installed
  (TensorFlow 2.10.1, OpenCV 4.8.1, NumPy 1.23.5, dlib 20.0.1).
- Downloaded dlib 68-point landmark model -> `team_ai_model/model/face_weights.dat`.
- `preprocessing/preprocess.py` self-test passes: tensor shape (1, 22, 80, 112, 3).
- **Lead first task** — `PRD.md` + `docs/workflow.md` (requirements + overall workflow).
- **Dhruv Sharma first task** — `team_video_processing/data_collection/collect.py` (collect video dataset).
- **Dipesh Yadav first task** — `team_ui/frontend/DESIGN.md` (basic UI design).
- Set up weekly Form-3 report tooling: `generate_report.py` + `.github/workflows/auto_weekly_report.yml`.
- Weekly logs now live in `docs/weekly/` — see [week-01.md](docs/weekly/week-01.md).

**Next sprint tasks:**
- Lead: dataset planning.
- Dhruv Sharma: video preprocessing.
- Dipesh Yadav: video upload interface (React + Vite + Tailwind scaffold).

**Blockers:** none.

---

## Task ownership — current "first tasks"

| Member | Sprint | First task | File |
|---|---|---|---|
| Harsh Soni (Lead) | AI Model Development | Project planning & requirements | `PRD.md`, `docs/workflow.md` |
| Dhruv Sharma | Video Processing & Data Preparation | Collect video dataset | `team_video_processing/data_collection/collect.py` |
| Dipesh Yadav | UI Integration & Development | Design user interface | `team_ui/frontend/DESIGN.md` |
