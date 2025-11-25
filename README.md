# BMW Multimodal Generative AI Challenge — Internship Coding Task

---

## Setup Instructions

### 1. Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify installation
```bash
python -m transformers-cli env
```
### 3. Folder layout
```bash
bmw_vlm_challenge/
├─ common/                      # Shared utilities (I/O, logging, viz)
│   ├─ io_utils.py
│   └─ viz.py
├─ task1_video_captioning/      # Task 1: VLM Captioning
├─ task2_preprocess_analysis/   # Task 2: Object + Scene Analysis
├─ task3_chatbot_integration/   # Task 3: Conceptual Chatbot Integration
├─ docs/                        # Extra notes and video link
│   ├─ VIDEO_LINK.txt
│   └─ REPORT_NOTES.md
└─ requirements.txt
```
---
## Core Dependencies

| Library | Purpose |
|----------|----------|
| `opencv-python` | Frame extraction from video |
| `torch`, `torchvision` | Core deep learning framework |
| `transformers` | BLIP captioning model (Hugging Face) |
| `pandas`, `pyarrow` | Data handling and tabular storage |
| `tqdm` | Progress tracking for loops |
| `ultralytics` | YOLOv8 object detection (Task 2) |
| `sentence-transformers`, `faiss-cpu` | Text embedding + retrieval (Task 3) |
| `pyyaml` | Configuration management |
| `matplotlib` | Visualization of analysis results |


---

## Running the Tasks

### Example — Task 1 Full Pipeline
```bash
python -m task1_video_captioning.run_task1 all \
    --cfg task1_video_captioning/configs/task1.yaml
```

### Example — Task 2

```bash
python -m task2_preprocess_analysis.run_task2 all \
    --cfg task2_preprocess_analysis/configs/task2.yaml
```

