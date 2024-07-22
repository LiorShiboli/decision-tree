# decision-tree

# Requirements

* conda (anaconda etc'...) - for manage the python envrioment

# Technologies

* poetry - for managing the python packages. Only python, pip and poetry are installed by conda.

# Install

```bash
conda env create -f environment.yml
conda activate decision-tree
poetry install
```

# Get Started

In terminal run the folowing commend for brute-force mode

```bash
python -m decision_tree brute-force
```

In terminal run the folowing commend for binary-entropy mode

```bash
python -m decision_tree binary-entropy
```

## CI (Formaters and Linters)

* note: the CI is not used because we dont have .py files in the project.

For running the Formaters for clean the code. Run in the terminal:

```bash
make fix
```

For running the Linters for check the code. Run in the terminal:

```bash
make lint
```

For running the Formaters and Linters for clean and check the code. Run in the terminal:

```bash
make fix-lint
```

List of the Formaters and Linters:

* black - for format the code
* isort - for sort and clean the imports
* flake8 - for check the code according to PEP8
* mypy - for static type checking


For removeing the cache. Run in the terminal:

```bash
make clean
```

In practice, you dont need to run this command, it called by the other make commands.

# License

MIT License

# Authors

* Lior Shiboli
* Omer Priel
