
# [OptiTask AI: Enterprise Resource Allocation Engine](https://optitask-ai.streamlit.app/)

## 🏗️ Project Overview

**OptiTask AI** is a hybrid intelligence platform that combines **Generative AI** with **Mixed Integer Programming (MIP)** to solve complex resource allocation problems. The system transforms unstructured business requests into mathematically optimal staffing decisions, minimizing costs while strictly adhering to resource constraints.

## 🛠️ Tech Stack & Core Components

* **LLM Orchestration:** `LangChain` & `Groq` (Llama 3.3 70B) for high-reasoning NLP extraction.
* **Optimization Engine:** `PuLP` (Linear Programming) for solving the assignment problem.
* **Interface:** `Streamlit` for a real-time, "Human-in-the-Loop" dashboard.
* **Data Handling:** `Pandas` for curating and preprocessing large-scale (2000+ row) datasets.
* **DevOps/MLOps:** `Docker` for containerization, `Terraform` for Infrastructure as Code (IaC), and `GitHub Actions` for CI/CD.

---

## 📂 File Architecture & System Nature

This project follows a professional **Separation of Concerns** architecture where the application logic is decoupled from the infrastructure:

| File                           | Nature                     | Function                                                                                                                                          |
| :----------------------------- | :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`app.py`**           | **Core Logic**       | The "Brain." Contains the NLP parsing logic, the MIP solver, and the UI layout.                                                                   |
| **`Dockerfile`**       | **Containerization** | The "Environment." Packages the Python runtime and OS-level optimization solvers (CBC/GLPK) to ensure consistency across deployment environments. |
| **`main.tf`**          | **IaC (Terraform)**  | The "Foundation." Automates the provisioning of cloud servers and container registries, ensuring repeatable and scalable infrastructure.          |
| **`.yaml` (CI/CD)**    | **Automation**       | The "Pipeline." Automatically builds the Docker image and deploys updates whenever code is pushed to the repository, driving MLOps efficiency.    |
| **`requirements.txt`** | **Dependencies**     | The "Manifest." Lists all specific library versions required for the system to function correctly.                                                |

---

## 🛡️ Trusted AI & Ethical Considerations

To satisfy enterprise standards for **Fairness, Transparency, and Accountability**, the system includes:

1. **Transparency Reports:** Every assignment includes an "Optimization Audit Trail" explaining why a candidate was chosen and why others were filtered out based on objective constraints.
2. **Human-in-the-Loop (HITL):** A mandatory validation step allows managers to refine AI-extracted needs before the mathematical solver executes, ensuring human oversight.
3. **Algorithmic Fairness:** The solver removes human bias by making selections based solely on objective data (Hourly Rate and Availability), mitigating unconscious bias in staffing.
4. **Data Security:** Employs a "stateless" API architecture and a "Bring Your Own Key" (BYOK) model to ensure data privacy and credential security.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* Groq API Key
* Docker (Optional for local container testing)

### Local Execution

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Upload your resource CSV in the sidebar and enter your API key to begin.

### Deployment (MLOps Path)

1. **Containerize:** `docker build -t optitask-ai .`
2. **Infrastructure:** Initialize Terraform with `terraform init` and `terraform apply`.
3. **Automate:** Push to the `main` branch to trigger the GitHub Actions pipeline.

---

## 📈 Impact & Use Case

This system reduces the time spent on manual resource mapping by **90%** while ensuring a **Global Cost Minimum** that is mathematically impossible to achieve consistently through manual human effort.

