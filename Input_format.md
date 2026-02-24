# MMGrader – Input JSON Format Specification

This document defines the required input JSON structure for MMGrader. Do check our [sample file](samples/sample3.json).

The input file must contain the following top-level keys:

- `questions`
- `answers`
- `concept_link`
- `concept_hierarchy`

---

# 1️⃣ Questions

The `questions` object contains all assessment questions.

Each question must include:

- `T` → Question text (string)
- `I` → Path to question image (string)
- `CL` → List of Concept Link IDs associated with this question (array of strings)

## Structure

```json
"questions": {
  "<question_id>": {
    "T": "Question text",
    "I": "path/to/image.png",
    "CL": ["concept_link_id"]
  }
}
```

## Example

```json
"questions": {
  "1": {
    "T": "Find the magnitude of vector A = (3,4) and explain how it relates to triangle law.",
    "I": "samples/sample.png",
    "CL": ["1"]
  }
}
```

---

# 2️⃣ Answers

The `answers` object contains all student responses.

Each student is identified by a unique student ID.

Each student:

- Answers multiple questions
- Each answer includes:
  - `T` → Answer text
  - `I` → Answer image path

## Structure

```json
"answers": {
  "<student_id>": {
    "<question_id>": {
      "T": "Student answer text",
      "I": "path/to/image.png"
    }
  }
}
```

## Example

```json
"answers": {
  "abc123": {
    "1": {
      "T": "Magnitude = 5 using sqrt(x^2 + y^2). Related to triangle sides.",
      "I": "samples/sample.png"
    }
  }
}
```

---

# 3️⃣ Concept Link

Defines grading rubrics and conceptual mappings.

Each concept link contains:

- `name` → Description of the concept connection
- `scoring_guide` → Rubric criteria
- `links` → Question IDs associated with this concept link

## Structure

```json
"concept_link": {
  "<concept_link_id>": {
    "name": "Concept name",
    "scoring_guide": {
      "<score>": "Description of grading criteria"
    },
    "links": ["question_id"]
  }
}
```

## Example

```json
"concept_link": {
  "1": {
    "name": "Magnitude connected to Triangle Law",
    "scoring_guide": {
      "1": "Correct magnitude formula",
      "2": "Correct computation",
      "3": "Connection to triangle representation"
    },
    "links": ["1", "3"]
  }
}
```

---

# 4️⃣ Concept Hierarchy

Defines high-level learning concepts used for conceptual aggregation.

Each hierarchy entry contains:

- `name` → Concept name

## Structure

```json
"concept_hierarchy": {
  "<concept_id>": {
    "name": "Concept Name"
  }
}
```

## Example

```json
"concept_hierarchy": {
  "1": {
    "name": "Magnitude of Vectors"
  },
  "2": {
    "name": "Direction of Vectors"
  }
}
```

---

# 📌 Complete Minimal Valid Example

```json
{
  "questions": {},
  "answers": {},
  "concept_link": {},
  "concept_hierarchy": {}
}
```

All four top-level keys are required.

---

# ⚠️ Validation Rules

- Question IDs must match between:
  - `questions`
  - `answers`
  - `concept_link.links`
- Concept link IDs in `questions.CL` must exist in `concept_link`
- Image paths must be valid
- Student IDs must be unique

---

# ✅ Summary

| Section           | Purpose                              |
| ----------------- | ------------------------------------ |
| questions         | Defines assessment questions         |
| answers           | Stores student responses             |
| concept_link      | Defines grading rubrics              |
| concept_hierarchy | Defines high-level learning concepts |

---

This structure enables MMGrader to automatically infer the quality of a student's mental model from their assessment.
