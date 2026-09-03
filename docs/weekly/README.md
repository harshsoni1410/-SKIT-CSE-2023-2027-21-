# Weekly workflow

The project is delivered in **weekly increments**. Each week:

1. Every member does that week's task **on their own branch**
   (`harsh`, `member1`, `member2`).
2. Each member commits with clear messages (these become the report's commit log).
3. By **Thursday** every member merges their branch into `main`.
4. The GitHub Actions workflow (`.github/workflows/auto_weekly_report.yml`) runs
   automatically Thursday 23:59 IST, generates the **Form-3 Weekly Progress Report**
   PDF from the git history, and commits it to `weekly_reports/`.
5. Download the PDF from `weekly_reports/` and submit it to the Lab Coordinator.

## Generate the report manually

```
venv\Scripts\activate
pip install reportlab matplotlib          # already in requirements.txt
python generate_report.py weekly          # last 7 days
python generate_report.py monthly         # last 30 days
python generate_report.py final           # whole project
```

The PDF is written to the repo root (git-ignored there); the workflow moves it into
`weekly_reports/`. The filename carries the project week number, e.g.
`..._Week-01_Weekly_Progress_Report_Form-3_2026-09-03.pdf` (Week 1 anchor Monday =
2026-08-31, set in `generate_report.py`).

## One-time GitHub setup

- Add the **mentor** and **Lab Coordinator** as repository collaborators.
- In the repo **Actions** tab, run *"Weekly Progress Report Auto-Commit"* once manually
  (`Run workflow`) to confirm it works.

## Week index

| Week | Dates | Focus | Log |
|---|---|---|---|
| 1 | 2026-09-03 → 2026-09-09 | First tasks: planning & requirements, data collection tool, UI design | [week-01.md](week-01.md) |
