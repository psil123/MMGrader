
# MMGrader : A framework to infer mental model of student's from multimodal assessments

STEM Mental models can play a critical role in assessing students’ conceptual understanding of a topic. They not only offer insights into what students know but also into how effectively they can apply, relate to, and integrate concepts across various contexts. Thus, students' responses are critical markers of the quality of their understanding and not entities that should be merely graded. However, inferring these mental models from student answers is challenging as it requires deep reasoning skills. We propose **MMGrader**, an approach that infers the quality of students' mental models from their multimodal responses using concept graphs as an analytical framework. In our evaluation with 9 openly available models, we found that the best-performing models fall short of human-level performance. This is because they only achieved an accuracy of approximately 40\%, a prediction error of 1.1 units, and a scoring distribution fairly aligned with human scoring patterns. With improved accuracy, these can be highly effective assistants to teachers in inferring the mental models of their entire classrooms, enabling them to do so efficiently and help improve their pedagogies more effectively by designing targeted help sessions and lectures that strengthen areas where students collectively demonstrate lower proficiency.

---

## Overview

MMGrader enables:

- 📚 Automatic grading of multimodal student responses
- 🖼️ Processing of both text and image inputs
- 🧠 Model-based reasoning for scoring
- 📊 Mental model graph generation for explainability
- 🔄 Support for multiple vision-language backends

The framework is modular, allowing easy integration of new multimodal models.

---

## Repository Structure

MMGrader/
│
├── main.py               # Entry point
├── models.py             # Model wrappers (Gemini, Qwen, LLaVA, etc.)
├── prompt.py             # Prompt templates
├── MentalModel.py        # Mental model generation & graph logic
├── helper.py             # Utility functions
├── requirements.txt      # Dependencies
│
└── samples/
    ├── sample1.json
    ├── sample2.json
    └── sample3.json

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/psil123/MMGrader.git
cd MMGrader
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Some models may require additional setup:

- HuggingFace authentication (`huggingface-cli login`)
- Google Gemini API key
- CUDA for GPU acceleration

---

## Usage

Run the sample script:

```bash
python main.py
```

The script will:

1. Load a sample JSON file
2. Initialize the selected model
3. Generate a mental model
4. Visualize the grading graph

---

## Input Format

Each sample JSON file should follow this structure:

```json
{
  "question": {
    "T": "Question text",
    "I": "path/to/question_image.jpg"
  },
  "answer": {
    "T": "Student answer text",
    "I": "path/to/answer_image.jpg"
  },
  "concept_link": "Concept description",
  "concept_link_score": "Scoring rubric"
}
```

Kindly refer to [Input_format.md](Input_format.md) .

---

## Supported Models

The `models.py` file provides wrappers for multiple multimodal models:

- Molmo
- LLaVA
- Qwen2-VL
- Gemini
- Gemma
- Pixtral
- Granite
- LLaMA Vision

Each model inherits from a shared abstract interface for consistency.

---

## Mental Model Module

`MentalModel.py`:

- Parses multimodal input
- Constructs structured representations
- Builds a mental models of students using concept graphs

## Requirements

Key libraries:

- torch
- transformers
- accelerate
- pillow
- google-generativeai
- tensorflow
- qwen-vl-utils
- vllm

See `requirements.txt` for full list.

---

## Notes

- Some models (e.g., Qwen2-VL) may require additional utility files such as `qwen_vl_utils.py`.
- Ensure correct API keys are set for proprietary models.
- GPU is recommended for large model inference.

---

## Citation

We will add the bibtex shortly. The paper has currently been accepted to EACL 2026 [Main Track] as a long paper.

Pritam Sil, Durgaprasad Karnam, Vinay Reddy Venumuddala and Pushpak Bhattacharyya. How effective are  VLMs in assisting humans in inferring the quality of mental models from Multimodal short answers? [EACL (Main) 2026]

---
