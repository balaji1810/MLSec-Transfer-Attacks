# Transfer-Attacks

by Birk and Balaji

## UV:

using uv:
```bash
uv run main.py
```
UV will automatically install the dependencies and run the code in a virtual environment.

If you add/remove dependencies without using UV, please add them to `project.toml` so that UV still work.

Otherwise, you can install the dependencies manually using whatever package manager you prefer,
by using the `requirements.txt` file or manually installing the packages listed in `project.toml`.
If you install the dependencies manually, make sure to install robustbench from git and not from pip!

## Folders:

```bash
.
├── data  # contains the datasets (auto-downloaded)
├── models  # contains model weights (auto-downloaded and trained ones)
└── src  # code
```


## links:

RubutstBench Model IDs:
https://github.com/RobustBench/robustbench?tab=readme-ov-file#cifar-10

