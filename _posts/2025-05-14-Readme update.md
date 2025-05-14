

---

## ✅ 보완된 `README.md` (최신 통합 완성본)

```markdown
# GUI Perception and Dataset Builder (UI-TARS Style)

This project converts GUI screenshots into structured, multi-task training data following the UI-TARS paper format.  
It supports OCR-based UI element localization and integrates LLM-based multi-task GUI understanding.

---

## ✨ Features

- 🔍 **OCR-based UI element detection (localizer)**
- 🧠 **Multi-task GUI understanding**: captioning, QA, marking, transitions
- 🤖 **GPT-4o integration** for dense task inference
- 🔁 **Legacy format converter** and unknown field filler
- 🧰 **Modular CLI interface**

---

## 🗂️ Project Structure

```

gui\_project/
├── main.py                          # Unified CLI entry point
├── model\_utils.py                   # GPT API + image base64 encoder
├── localizer.py                     # OCR-based UI element extractor
├── run\_localizer.py                 # (optional) standalone batch localizer
├── aw\_data\_converter/
│   └── data\_utils.py                # Format converter & GPT unknown filler
├── gui\_tasks/
│   ├── base\_task.py
│   ├── element\_description\_task.py
│   ├── dense\_captioning\_task.py
│   ├── qa\_task.py
│   ├── set\_of\_mark\_task.py
│   └── state\_transition\_captioning\_task.py

````

---

## ⚙️ Installation

```bash
pip install pytesseract python-dotenv requests pillow
````

> ⚠️ Requires [Tesseract OCR engine](https://github.com/tesseract-ocr/tesseract)
>
> * Ubuntu: `sudo apt install tesseract-ocr`
> * macOS: `brew install tesseract`
> * Windows: download installer from GitHub

---

## 🔐 .env Configuration

```env
API_KEY=your_openai_api_key
API_ENDPOINT=http://your.gpt4o.server:9000/chat/completions
MODEL_NAME=gpt-4o
```

---

## 🚀 CLI Usage (via `main.py`)

### 🧭 1. OCR-based UI Element Localization Only

```bash
python main.py localize \
  --image_dir screenshots \
  --output_dir som_output \
  --json_dir element_jsons
```

#### ➤ Output:

* `som_output/` → Marked screenshots (SOM) with red boxes and ID numbers
* `element_jsons/` → `localized_<name>.json` with element list:

```json
{
  "id": 1,
  "name": "Submit",
  "bbox": [120, 300, 280, 350]
}
```

---

### 🤖 2. Run Multi-task Perception (with or without localizer)

Run all tasks:

```bash
python main.py run \
  --image_dir screenshots \
  --all \
  --output gui_results.json \
  --use_localizer
```

Run only one task:

```bash
python main.py run \
  --image screenshots/home.png \
  --task qa \
  --question "What does this button do?" \
  --output qa_home.json \
  --use_localizer
```

---

### 🔁 3. Convert Legacy Dataset

```bash
python main.py convert \
  --input old_format.json \
  --output converted.json
```

---

### 🧠 4. Fill Unknown Fields with GPT

```bash
python main.py fill \
  --input converted.json \
  --image_dir screenshots \
  --output filled.json
```

---

## 📷 Marked Screenshot (SOM)

* **SOM image**: Red box for each UI element + ID number
* **Element JSON**:

```json
{
  "id": 2,
  "name": "Cancel",
  "bbox": [300, 300, 480, 350]
}
```

---

## 🧩 Supported Tasks

| Task                          | Description                                |
| ----------------------------- | ------------------------------------------ |
| `element_description`         | Structure UI elements into JSON            |
| `dense_captioning`            | Generate caption for entire screen         |
| `qa`                          | Answer screen-level or element-specific Qs |
| `set_of_mark`                 | Identify visually emphasized components    |
| `state_transition_captioning` | Describe differences between two screens   |

---

## 🧰 Developer Tips

* All tasks are modular via `BaseGUITask`
* Add new tasks easily in `gui_tasks/`
* `localizer.py` is fully independent and reusable
* You can also use `run_localizer.py` as standalone

---

## 📄 License

MIT License

---

## ✍️ Author

Created by \[Your Name].
Contributions welcome!

```

---
```
