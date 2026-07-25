# 🤖 Multi-Agent Coding System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous multi‑agent system that solves coding problems using a **Test‑Driven Development (TDD)** workflow. Built with **LangGraph**, it coordinates four specialist agents – **Planner**, **Test Architect**, **Implementer**, and **Reviewer** – to generate, verify, and iteratively improve code until it passes all tests.

**Key innovation**: The system uses a **deterministic execution gate** (not an LLM) to validate code, ensuring that feedback is grounded in actual test results, not hallucinated expectations. This makes the system reliable even with weaker models.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Configuration](#️-configuration)
- [🎯 Usage](#-usage)
- [🧠 Why Multi‑Agent?](#-why-multiagent)
- [🛠️ Customisation](#️-customisation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- **4‑Agent TDD Workflow**
  - **Planner** – writes a human‑readable specification (`spec.md`)
  - **Test Architect** – generates a brute‑force reference and fuzz tests (`test_solution.py`)
  - **Implementer** – writes the solution (`solution.py`)
  - **Reviewer** – translates test failures into actionable fix instructions

- **LangGraph Orchestration**
  - Stateful graph with checkpointing (resume failed runs)
  - Conditional routing: distinguishes **test‑harness bugs** from **solution bugs**
  - Automatic retry loop (max 3 attempts) with early exit on success

- **Model‑Agnostic & Cost‑Efficient**
  - Assign different models per agent (e.g., Gemini for Implementer, Llama 8B for others)
  - Supports OpenAI, Groq, Gemini, DeepSeek, Ollama, and any OpenAI‑compatible endpoint

- **Safe Execution Environment**
  - Blocks `git` commands (prevents accidental commits)
  - Runs tests in isolated temporary directories

- **Built‑in Evaluation**
  - Hidden test harness for benchmarking (DSA and logic problems)
  - Results saved as JSON with metadata

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Planner] --> B[Test Architect]
    B --> C[Syntax Validator]
    C -->|pass| D[Implementer]
    C -->|fail| B
    D --> E[Test Runner]
    E -->|pass| F[Done]
    E -->|fail| G{classify traceback}
    G -->|solution error| H[Reviewer]
    H --> D
    G -->|test error| B
