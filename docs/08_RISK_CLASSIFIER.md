# 08 — Risk Classifier (the "trained model" for the project)

## 1. Purpose

A small text classifier that reads a planned action (tool + normalised args + short context) and predicts a risk label:
`safe`, `sensitive`, or `dangerous`. It is an **extra safety layer**: it can only **raise** the tier decided by the rules,
never lower it (claim C4). It runs locally on CPU (ONNX), adds no API calls, and needs no internet.

Why this is a good final-year "model": it is small, trainable free on Google Colab, measurable (precision/recall per class),
and directly connected to a real safety problem. It also gives you a solid answer to "what did YOU train?".

## 2. Label definitions and tier mapping

| Label | Meaning | Examples | Minimum tier applied |
|-------|---------|----------|----------------------|
| `safe` | Read-only or trivially reversible | open notepad, google search, system_info, list folder | 0 |
| `sensitive` | Changes files/state or communicates | create/overwrite file, type into app, dictation, lock PC | 1 |
| `dangerous` | Deletion, messaging others, security-relevant, unclear/destructive intent | delete files, send WhatsApp, anything mentioning disabling protections or credentials | 2 |

`final_tier = max(rule_tier, classifier_min_tier)`. Tier 3 stays **rules only** (hard blocks are never delegated to a model).
Low-confidence predictions (max prob < threshold) are treated as one label more severe (fail toward caution).

## 3. Input format

One string per action, produced by a deterministic serializer (`ml/serialize.py`) so training and inference match:

```
tool=delete_path | args={"paths":["<PATH:workspace/report.docx>"]} | user="delete the old report" | tainted=false
```

- Replace real paths/phones with placeholders (`<PATH:…>`, `<PHONE>`) to reduce overfitting and protect privacy.
- Include the user's original command text and the `tainted` flag: injection-like phrasing is a key signal.

## 4. Dataset

**Target size:** ≥ 3,000 labelled examples (more is better), split 70/15/15 train/val/test, stratified by label **and** by tool.
**Never** let near-duplicates cross splits (dedupe by normalised string before splitting).

**How to build**
1. Programmatic generation: templates × tools × argument variations × paraphrases of user commands (covers all registered tools).
2. LLM-assisted generation: prompt a strong LLM for varied user commands and edge cases per tool and label (see prompt below). Ask for JSON lines.
3. Adversarial examples (≥ 15% of data): injection phrasing, politeness/urgency tricks, mixed-language (English/Hinglish if you plan to support it), typos, misleading descriptions ("just a small cleanup" that deletes many files).
4. **Hand-review at least 500 examples** (all of the test set!) and fix labels. Record inter-check notes. The test set must be clean; that is what examiners will trust.
5. Store as `training/data/{train,val,test}.jsonl` with fields: `id, text, label, tool, source (template|llm|manual), adversarial (bool)`.

**Generation prompt (use in any strong LLM chat, in batches):**
```
You are generating training data for a risk classifier for a Windows automation agent.
Tools: <paste tool names + descriptions from docs/04_TOOLS_SPEC.md>.
Labels: safe, sensitive, dangerous (definitions: <paste table from section 2>).
Produce 40 JSON lines with fields {"user": "...", "tool": "...", "args": {...}, "label": "..."}.
Cover: ordinary commands, typos, indirect wording, urgency/pressure wording, attempts to hide a destructive action
inside a harmless-sounding request, and ambiguous requests. Use placeholder paths like <PATH:workspace/x.txt>.
Balance labels roughly 40/30/30. Do not repeat earlier examples.
```

## 5. Model and training

- Base model: a small pre-trained encoder (DistilBERT-class or MiniLM-class; e.g. `distilbert-base-uncased` or `microsoft/MiniLM-L12-H384-uncased`). Verify current availability on Hugging Face before starting.
- Framework: Hugging Face `transformers` + `datasets` + `evaluate` on **Google Colab (free GPU)**. Keep the notebook in `training/train_risk_classifier.ipynb`.
- Hyperparameters to start: max_len 128, batch 32, lr 2e-5–5e-5, 3–5 epochs, weight decay 0.01, class weights if imbalanced, seed fixed (report seeds, run ≥ 3 seeds for mean ± std).
- Metrics: accuracy, macro-F1, **per-class precision/recall**, confusion matrix. Priority: **recall on `sensitive` and `dangerous`** (missing a risky action is worse than over-warning). Tune the decision threshold on the validation set to reach the recall target (e.g., ≥ 0.98 for `dangerous`) and report the resulting precision cost.
- Baselines to compare (table in report): (a) rules only, (b) keyword/TF-IDF + logistic regression, (c) LLM-as-judge zero-shot, (d) your fine-tuned model.
- Robustness: evaluate separately on the adversarial subset; report the gap.
- Export: `optimum` or `torch.onnx.export` → ONNX → dynamic int8 quantisation (`onnxruntime.quantization`). Compare accuracy before/after quantisation and CPU latency (target < 30 ms per action on a laptop CPU).
- Package: `src/jarvis/ml/models/risk_classifier/{model.onnx, tokenizer files, labels.json, threshold.json}`. Keep the model file under ~100 MB or download on first use from a release you host; never commit large files to Git without Git LFS.

## 6. Integration (`src/jarvis/ml/risk.py`, Phase 8)

```python
class RiskClassifier:
    def __init__(self, model_dir: Path, threshold: float): ...
    def predict(self, step: Step, user_input: str, tainted: bool) -> tuple[Label, float]: ...
    def min_tier(self, step, user_input, tainted) -> int:
        label, p = self.predict(...)
        if p < self.threshold:
            label = more_severe(label)
        return {"safe": 0, "sensitive": 1, "dangerous": 2}[label]
```
- Load lazily with `onnxruntime` (CPU provider) and the `tokenizers`/`transformers` tokenizer only (avoid importing torch at runtime).
- If the model is missing or fails to load → log a warning and continue with rules only (never crash the agent).
- Test: a property test proving `decide()` never returns a tier lower than the rules-only tier when the classifier is enabled.

## 7. Evaluation for the report

1. Test-set table: per-class precision/recall/F1, macro-F1, confusion matrix, quantised vs full model.
2. Adversarial-subset results.
3. Baseline comparison (rules-only vs TF-IDF+LR vs LLM-judge vs fine-tuned).
4. End-to-end: red-team suite with and without the classifier (extra actions escalated; false-block increase).
5. Error analysis: 10 misclassified examples with explanations and what you would change.
6. Ethics/limits: dataset is synthetic; classifier is advisory and can only escalate; rules remain the source of truth.

## 8. Alternative "model" options (if your college prefers a different one)

- **Tool-routing classifier**: predicts which layer to use (URL launch / accessibility tree / DOM / vision fallback) from the request text.
- **Custom wake word** ("Hey Jarvis" variants or your own phrase) with openWakeWord's training pipeline.
- **Intent classifier** to skip the LLM for very common commands (cheaper and faster; measure the accuracy/latency trade-off).

Pick one main model and do it properly rather than three superficially. The risk classifier is recommended.
