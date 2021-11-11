# Experiments repo

## Setup


1. Install [pyenv](https://github.com/pyenv/pyenv) on your machine if you didn't do it already.
```
2. Install latest Python 3.8 if you don't have it already:
```bash
pyenv install 3.8.11
echo "3.8.11" > ~/.pyenv/version
```
3. Install [Poetry](https://python-poetry.org) if you don't have it already:
```bash
pip install poetry
```
4. Clone this repository.

5. Create a virtual environment and install required packages using:
```bash
poetry install
```
6. Install pre-commit hooks using:
```bash
poetry run pre-commit install
```