
---

## 📘 README.md

```markdown
# GUI Perception and Dataset Builder (UI-TARS Style)

This project enables automatic analysis, transformation, and enhancement of GUI screenshots into structured, multi-task training data—aligned with the UI-TARS paper format. It supports:

- ✨ Multi-task GUI perception (captioning, element detection, QA, etc.)
- 📂 Bulk conversion from existing datasets
- 🤖 Filling incomplete data via GPT-4o vision models
- 📦 Modular architecture for maintainability and scalability

---

## 📁 Project Structure

```

gui\_project/
├── main.py                         # Entry point for all operations
├── model\_utils.py                  # GPT-4o API + image encoding logic
├── aw\_data\_converter/
│   ├── **init**.py
│   └── data\_utils.py              # Old → new format converter & answer filler
├── gui\_tasks/
│   ├── base\_task.py
│   ├── element\_description\_task.py
│   ├── dense\_captioning\_task.py
│   ├── qa\_task.py
│   ├── set\_of\_mark\_task.py
│   └── state\_transition\_captioning\_task.py

````

---

## 🚀 Quick Start

### 🔹 1. Convert Legacy Format

```bash
python main.py convert \
  --input old_format.json \
  --output converted_data.json
````

### 🔹 2. Fill Unknown Answers using GPT-4o

```bash
python main.py fill \
  --input converted_data.json \
  --image_dir screenshots/ \
  --output completed_data.json
```

### 🔹 3. Run Perception Tasks (Single or Batch)

```bash
python main.py run \
  --image_dir screenshots/ \
  --all \
  --output gui_result.json
```

Or run only one task:

```bash
python main.py run \
  --image screenshots/home.png \
  --task qa \
  --question "What is the purpose of this screen?" \
  --output result_qa.json
```

---

## 🧠 Supported Tasks

* `element_description` – bounding boxes, types, visuals, functions
* `dense_captioning` – natural language overview
* `qa` – screen-level question answering
* `set_of_mark` – identify highlighted/selected elements
* `state_transition_captioning` – explain UI state changes between two screenshots

---

## 🔐 .env Configuration

Create a `.env` file in the root with the following:

```env
API_KEY=your_openai_api_key
API_ENDPOINT=http://your.gpt4o.server/chat/completions
MODEL_NAME=gpt-4o
```

---

## 🛠️ Requirements

```bash
pip install python-dotenv requests
```

---

## 🧩 Extending the Project

* Add new data converters to `aw_data_converter/`
* Add new GUI tasks by extending `BaseGUITask` in `gui_tasks/`
* Easily adapt to other LLM APIs by modifying `model_utils.py`

---

## 📄 License

MIT License

---

## ✍️ Author

Created by \[Your Name]. Contributions welcome.

```

---

