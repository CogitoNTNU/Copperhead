# Copperhead

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/Copperhead/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/Copperhead)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/Copperhead)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/copperhead-ikon.png" width="50%" alt="Copperhead logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b>📋 Table of contents </b></summary>

- [Copperhead](#copperhead)
  - [Description](#description)
  - [🛠️ Prerequisites](#%EF%B8%8F-prerequisites)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [📖 Generate Documentation Site](#-generate-documentation-site)
  - [Testing](#testing)
  - [Team](#team)
    - [License](#license)

</details>

## Description

Copperhead is a Cogito NTNU project about teaching agents to play Snake with reinforcement learning. The shared library `coppercore` (in [packages/coppercore](packages/coppercore)) contains the Snake environment, replay buffers, DQN agents and training utilities. Each team member's experiments live under [projects/](projects).

## 🛠️ Prerequisites

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

## Getting started

1. **Clone the repository**:

   ```sh
   git clone https://github.com/CogitoNTNU/Copperhead.git
   cd Copperhead
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

1. **Set up pre commit** (only for development):

   ```sh
   uv run pre-commit install
   ```

## Usage

To watch a random agent play Snake, run the following command from the root directory of the project:

```bash
uv run python -m coppercore.dqn.run_random_snake
```

See the [Snake environment README](packages/coppercore/src/coppercore/dqn/README.md) for how to use the environment with your own agent.

### 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the lastes commit on main by viewing the `gh-pages` branch on GitHub: [https://cogitontnu.github.io/Copperhead/](https://cogitontnu.github.io/Copperhead/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

## Team

This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for the time and effort you have put into making this project a reality.

### License

______________________________________________________________________

Distributed under the MIT License. See `LICENSE` for more information.
