# 🤖 Multi-Agent TDD Coding System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous multi‑agent system that solves coding problems using a **Test‑Driven Development (TDD)** workflow with **memory, self-correction, and quality gates**. Built with **LangGraph**, it coordinates five specialist agents – **Planner**, **Test Architect**, **Implementer**, **Reviewer**, and **Critic** – to generate, verify, and iteratively improve code until it passes all tests.

**Key innovation**: The system uses a **deterministic execution gate** (not an LLM) to validate code, a **Memory Bank** to learn from past mistakes, and a **Code Quality Loop** to self-correct before test execution. This makes the system reliable with **ANY model** – even Llama 8B – while being **cost‑efficient** by using strong models only where they matter.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🧠 Why Multi‑Agent?](#-why-multiagent)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Configuration](#️-configuration)
- [🎯 Usage](#-usage)
- [🛠️ Customisation](#️-customisation)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

### 🧠 5‑Agent TDD Workflow
- **Planner** – writes a human‑readable specification (`spec.md`)
- **Test Architect** – generates a simple, correct brute‑force reference and fuzz tests (`test_solution.py`)
- **Implementer** – writes the solution (`solution.py`) using the STANDARD algorithm
- **Reviewer** – translates test failures into precise, actionable JSON fixes
- **Critic** – reviews code quality BEFORE the test runner (optional)

### 🔁 Self‑Correction & Learning
- **Memory Bank** – stores every attempt (code + error + fix) and summarizes it for the Implementer
- **Code Quality Loop** – catches syntax errors via `ast.parse()` BEFORE the test runner
- **Refinement Node** – fixes ONLY the specific issues flagged by the Code Checker or Critic
- **Intelligent Routing** – distinguishes **test‑harness bugs** from **solution bugs** and routes to the right agent

### 💰 Cost‑Efficient Model Strategy
- **Strong model (Gemini/GPT‑4o)** for the **Implementer** (heavy lifting)
- **Cheap models (Groq/Llama 8B)** for Planner, Test Architect, Reviewer, and Critic
- **~75% of calls are cheap** – massive cost savings

### 🔒 Safe Execution Environment
- Blocks `git` commands (prevents accidental commits)
- Runs tests in isolated temporary directories
- Deterministic validation gates (no LLM hallucinations)

### 📊 Built‑in Evaluation
- Hidden test harness for benchmarking (DSA and logic problems)
- Results saved as JSON with full metadata
- Automatic success/failure logging

---

## 🏗️ Architecture

### Full System Flow

```mermaid
graph TD
    A[Planner] --> B[Test Architect]
    B --> C[Syntax Validator]
    C -->|pass| D[Memory Summarizer]
    C -->|fail| B
    D --> E[Implementer]
    E --> F[Code Checker]
    F -->|syntax issues| G[Refinement]
    G --> F
    F -->|clean| H[Critic]
    H -->|issues| G
    H -->|approved| I[Test Runner]
    I -->|pass| J[Done]
    I -->|fail| K{classify traceback}
    K -->|test error| B
    K -->|solution error| L[Reviewer]
    L --> M[Memory Bank]
    M --> D
    I -->|max retries| J
    
    style D fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style M fill:#f96,stroke:#333
