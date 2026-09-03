# LipSense — Team Plan & Sprints

3 members, 3 sprints. Dates are DD/MM/YY.

---

## TEAM LEAD — Harsh Soni
### Sprint: AI Model Development — folder: [`team_ai_model/`](../team_ai_model/)

| # | User Story | Start | End | Details |
|---|---|---|---|---|
| 1 | Project planning & requirements | 10/08/26 | 20/08/26 | Finalized project requirements and overall workflow. |
| 2 | Dataset planning | 21/08/26 | 15/09/26 | Planned dataset collection and preprocessing approach. |
| 3 | AI model design | 16/09/26 | 30/09/26 | Designed AI model architecture for lip reading. |
| 4 | Model development | 01/10/26 | 31/10/26 | Developed the initial AI model and training pipeline. |
| 5 | Model training | 01/11/26 | 30/11/26 | Trained and optimized the AI model. |
| 6 | System integration | 01/12/26 | 31/01/27 | Integrated the trained model with the complete system. |
| 7 | Testing & deployment | 01/02/27 | 15/03/27 | Performed testing, optimization and final deployment. |

---

## MEMBER 1 — Video Processing
### Sprint: Video Processing & Data Preparation — folder: [`team_video_processing/`](../team_video_processing/)

| # | User Story | Start | End | Details |
|---|---|---|---|---|
| 1 | Collect video dataset | 10/08/26 | 15/08/26 | Collected and organized suitable speech video data. |
| 2 | Video preprocessing | 16/08/26 | 31/08/26 | Preprocessed videos and prepared input frames. |
| 3 | Face detection | 01/09/26 | 15/09/26 | Implemented face detection from video frames. |
| 4 | Lip region detection | 16/09/26 | 30/09/26 | Detected and extracted the lip region from faces. |
| 5 | Data normalization | 01/10/26 | 31/10/26 | Normalized and prepared extracted lip data. |
| 6 | Dataset preparation | 01/11/26 | 30/11/26 | Prepared labeled data for AI model training. |
| 7 | Video processing integration | 01/12/26 | 15/03/27 | Integrated video processing module with the AI system. |

---

## MEMBER 2 — UI Development
### Sprint: UI Integration & Development — folder: [`team_ui/`](../team_ui/)

| # | User Story | Start | End | Details |
|---|---|---|---|---|
| 1 | Design user interface | 10/08/26 | 20/08/26 | Designed the basic interface for the application. |
| 2 | Video upload interface | 21/08/26 | 15/09/26 | Developed interface for uploading speech videos. |
| 3 | Video display | 16/09/26 | 30/09/26 | Added video preview and playback functionality. |
| 4 | Prediction display | 01/10/26 | 31/10/26 | Added predicted output and confidence display. |
| 5 | Result history | 01/11/26 | 30/11/26 | Added interface for viewing previous results. |
| 6 | Model integration | 01/12/26 | 31/01/27 | Connected the UI with the AI model and backend. |
| 7 | UI testing & improvement | 01/02/27 | 15/03/27 | Tested and improved the complete user interface. |

---

## Current focus — "first tasks"

| Member | Task | Deliverable |
|---|---|---|
| Harsh (Lead) | Project planning & requirements | [PRD.md](../PRD.md) + [workflow.md](workflow.md) |
| Member 1 | Collect video dataset | `team_video_processing/data_collection/collect.py` |
| Member 2 | Design user interface | `team_ui/frontend/DESIGN.md` |

## Shared rules

1. **One preprocessing module** — `team_video_processing/preprocessing/`. Everyone imports from it.
2. **Class order** comes from dataset folder names, never hardcoded.
3. Update [PROGRESS.md](../PROGRESS.md) after each commit.
4. The model input shape in [PRD.md § 7](../PRD.md) is not changed without team discussion.
