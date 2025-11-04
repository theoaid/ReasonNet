# ReasonNet: Toward Reasoning-Guided Domain Code Generation

## Overview

ReasonNet is a research project that investigates the effectiveness of different reasoning approaches in generating domain-specific code. The project specifically focuses on remote sensing and geographic information systems (GIS) tasks, using NDVI (Normalized Difference Vegetation Index) computation as a test case to evaluate how different levels of reasoning guidance affect code quality.

## Research Question

How do different reasoning approaches impact the quality of generated code for domain-specific tasks? The project compares four distinct approaches:

1. **No Reasoning**: Direct code generation without any contextual reasoning
2. **Document-Level Reasoning**: Using comprehensive domain documentation as context
3. **Section-Level Reasoning**: Using targeted sections of domain documentation
4. **Manual Reasoning**: Human-crafted reasoning snippets for each task

## Methodology

The project uses a systematic evaluation approach:

- **Domain Focus**: Remote sensing tasks involving NDVI calculation and land cover classification
- **Test Cases**: 8 different NDVI-related programming tasks of varying complexity
- **Evaluation**: Human annotators ranked code outputs from different reasoning approaches
- **Models**: Large Language Models (LLMs) used for both reasoning generation and code synthesis

### Reasoning Pipeline

The `ReasonNetWithSectionsNDocs` class in `src/reason_net_sections_n_docs_with_ranker.py` implements the core reasoning pipeline:

1. **Domain Classification**: Automatically identifies the scientific domain of the user request
2. **Reasoning Generation**: Creates contextual reasoning snippets based on the chosen approach
3. **Library Suggestion**: Recommends appropriate Python libraries for the domain
4. **Code Generation**: Produces the final code using the structured reasoning plan

## Key Findings

Based on the annotator preferences analysis (see `results/annotator_preferences.png` and individual annotator outputs `results/output_annotator_X.png`):

### Main Results

- **Manual Reasoning** consistently received the highest ratings across annotators
- **Section-Level Reasoning** showed competitive performance, often ranking second
- **No Reasoning** generally produced lower-quality code outputs
- **Document-Level Reasoning** showed mixed results depending on the specific task complexity

### Performance Insights

The analysis reveals that:

1. **Context Quality Matters**: More focused reasoning (manual and section-level) outperforms broad or absent context
2. **Task Complexity**: Simple NDVI calculations benefit less from reasoning than complex GIS operations
3. **Annotator Consistency**: Different annotators showed similar preferences, validating the approach
4. **Domain Knowledge**: Reasoning that incorporates domain-specific knowledge significantly improves code quality

## Project Structure

```
├── src/                          # Core source code
│   ├── reason_net_sections_n_docs_with_ranker.py  # Main reasoning pipeline
│   ├── manually_crafted_reasoning.json            # Human-authored reasoning examples
│   ├── no_reasoning.json                         # Baseline approach data
│   ├── model_triples.json                       # Model configuration
│   └── requirements.txt                         # Dependencies
├── notebook/
│   └── Analysis_of_results.ipynb               # Results analysis and visualization
├── annotations/                                 # Human evaluation data
│   └── participants/                           # Annotator responses
├── results/                                    # Generated visualizations and outputs
└── reasonvenv/                                # Python virtual environment
```

## Running the Project

### Prerequisites

- Python 3.10+
- Access to Hugging Face models (requires API key in `hf_key.txt`)
- Computational resources (GPU recommended for model inference)

### Local Setup

1. **Clone and Setup Environment**:
```bash
cd ReasonNet
python -m venv reasonvenv
source reasonvenv/bin/activate  # On macOS/Linux
pip install -r src/requirements.txt
```

2. **Configure Hugging Face Access**:
```bash
# Add your HF API key to hf_key.txt
echo "your_huggingface_api_key" > hf_key.txt
```

3. **Run the Reasoning Pipeline**:
```bash
cd src
python reason_net_sections_n_docs_with_ranker.py
```

### Cluster Deployment

This project was originally run on a computational cluster for large-scale model inference. For cluster deployment:

- Ensure GPU access for model loading and inference
- Configure distributed computing if using multiple nodes
- Adjust batch sizes and memory settings in the model configuration
- Set up proper environment variables for cache directories

### Analyzing Results

To reproduce the analysis:

```bash
cd notebook
jupyter notebook Analysis_of_results.ipynb
```

This notebook generates the visualizations found in the `results/` directory, including:
- Annotator preference heatmaps
- Performance comparisons across reasoning approaches
- Word clouds showing reasoning content analysis

## Code Examples

The project generates Python code for various NDVI-related tasks. Example domains include:

- Computing NDVI from reflectance values
- Classifying land cover based on NDVI thresholds
- Processing GeoTIFF satellite imagery
- Handling edge cases (division by zero, missing data)

Each reasoning approach produces different code patterns and incorporates varying levels of domain expertise and error handling.

## Future Work

The ReasonNet framework can be extended to:
- Other scientific domains beyond remote sensing
- Different programming languages
- More sophisticated reasoning architectures
- Integration with domain-specific knowledge bases

## Contributing

This research codebase demonstrates the methodology and findings. The core components in `src/` can be adapted for similar domain-specific code generation research.