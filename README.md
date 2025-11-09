# DeepKEGG-Agent
![Uploading 新建 PPT 演示文稿 (2)_20251030211618_03(1)(1).png…]()


[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Interactive AI Agent for Automated Multi-Omics Cancer Outcome Prediction.

---

## 📖 Abstract

Accurate prediction of cancer outcomes from multi-omics data is essential for advancing precision oncology but remains constrained by complex scripting and high technical barriers. To address this challenge, we present **DeepKEGG-Agent**, an interactive framework that automates the entire workflow for multi-omics-based cancer outcome prediction through a dialogue-driven paradigm. A large language model (LLM) serves as the core agent, orchestrating data loading, feature integration, model training, and results evaluation. Users can specify datasets, select models, and configure experiments through a natural language interface without writing code. The system supports both conventional machine learning methods and pathway-informed deep models, including a graph neural network (`DeepKEGG_v2`) that models pathway-level interactions guided by KEGG knowledge. By lowering technical barriers and enhancing reproducibility, DeepKEGG-Agent makes complex multi-omics modeling more accessible to a broader range of biomedical researchers.

## ✨ Key Features

- **💬 Dialogue-Driven Workflow**: Control the entire modeling pipeline using simple, natural language commands. No coding required.
- **🤖 Automated End-to-End Pipeline**: Automates all stages from data loading and validation to model training, evaluation, and reporting.
- **🧩 Flexible Model Support**:
  - **Pre-defined Models**: Out-of-the-box support for specialized models like `DeepKEGG`, `LSTM`, and `Transformer`.
  - **Classic ML via LLM**: Dynamically generates `scikit-learn` pipelines for models like `SVM`, `LR`, and `RandomForest` on the fly.
  - **Dynamic Model Design**: A powerful `design` command allows the agent to generate novel, complex model architectures (e.g., GNNs) from a high-level user description.
- **🧬 Biology-Informed Modeling**: Seamlessly integrates biological prior knowledge, such as KEGG pathways, into deep learning models.
- **🔄 Reproducibility**: Automatically generates configuration files (`.json`) for each run, ensuring experiments are fully reproducible.

## 🏗️ System Architecture

The framework is structured into four primary stages, designed to systematically translate high-level user intent into concrete computational results.

 ![Uploading 论文流程图_20251102170858_05(1).png…]()



1.  **Environment Configuration**: Initializes the system by loading LLM configurations.
2.  **Conversational Task Initialization**: An LLM-powered planner parses user instructions and formulates an executable plan.
3.  **Adaptive Model Selection, Tuning, and Testing**: A model dispatcher routes the task to the appropriate training function, whether for pre-defined, LLM-generated, or dynamically designed models.
4.  **Experiment Result Summarizer**: Aggregates results from all experiments into a central CSV file and generates final reports.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd DeepKEGG-Agent
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project relies on several core libraries. Install them using pip.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some models, like Mamba or certain GNN layers, may require special dependencies (`mamba-ssm`, `torch-geometric`). Please follow their official installation guides if you intend to use them.*

4.  **Set up Environment Variables:**
    The agent requires an LLM API key to function. You can set this in your terminal before launching the agent.
    ```bash
    # For DeepSeek
    export LLM_PROVIDER="deepseek"
    export DEEPSEEK_API_KEY="sk-..."

    # For OpenAI
    export LLM_PROVIDER="openai"
    export OPENAI_API_KEY="sk-..."
    ```
    The data directory is assumed to be at `/tmp/DeepKEGG-master`. You can change this by setting the `BASE_DIR` environment variable.

## ⚡ Quick Start

The primary entry point is `chat_cli.py`, an interactive command-line interface.

1.  **Launch the Agent:**
    Navigate to the `agent` directory and run:
    ```bash
    cd agent
    python chat_cli.py
    ```

2.  **Run a Pre-defined Model (e.g., DeepKEGG):**
    The agent will greet you. At the `>` prompt, simply state your goal in natural language.

    ```
    > nl run DeepKEGG on LIHC with all omics
    ```
    The agent will parse your request and show a configuration summary. If it looks correct, start the training:
    ```
    > train
    ```

## 🌟 Advanced Usage: Dynamic Model Design

This is the most powerful feature of DeepKEGG-Agent. You can ask the agent to invent a new model for you.

1.  **Design a New Model:**
    Use the `design` command, followed by a name for your new model and a description of your idea.

    ```
    > design DeepKEGG_v2 "design a graph neural network that models pathway-level interactions"
    ```
    The agent will use its "Planner" and "Coder" capabilities to generate the complete Python code for this new model and save it in `agent/generated_models/`.

2.  **Use the New Model:**
    Once designed, you can immediately use your new model by its name in the `nl` command.

    ```
    > nl use DeepKEGG_v2 for LIHC with all omics
    ```

3.  **Train the New Model:**
    ```
    > train
    ```
    The system will dynamically load your newly created `DeepKEGG_v2` model and start the training process.

## 📁 Project Structure

```
.
├── agent/
│   ├── models/                 # Stores pre-defined model architectures (LSTM, Transformer, etc.)
│   ├── generated_models/       # Directory where AI-generated models are saved
│   ├── chat_cli.py             # Main interactive command-line interface
│   ├── step1_design.py         # Core script for the "design" workflow
│   ├── step2_nl.py             # Parses natural language into JSON configuration
│   ├── step3_run.py            # Executes the training and evaluation pipeline
│   └── step5_report.py         # Generates the final HTML report
├── configs/
│   └── last_run_config.json    # The configuration file for the last run
├── data/                       # Placeholder for data (actual data is at /tmp/DeepKEGG-master)
├── runs/
│   ├── all_models_metrics.csv  # Global summary table of all experiment results
│   └── ...                     # Individual run directories with detailed results
└── README.md                   # This file
```

## 🎓 Citing Our Work

If you use DeepKEGG-Agent in your research, please cite our paper:

```bibtex
@article{YourName2025DeepKEGGAgent,
  title={DeepKEGG-Agent: An Interactive AI Agent for Automated Multi-Omics Cancer Outcome Prediction},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025},
  volume={XX},
  pages={XX-XX}
}
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
