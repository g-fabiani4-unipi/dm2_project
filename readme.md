# Repository for Data Mining Advanced Topics and Applications project
## Directories
- `data`: raw data;
- `src`: the project source code (jupyter notebooks, scripts etc.);
- `results`: intermediate data generated during cleanup and analysis as well as final results such as figures and tables;
- `doc`: documents associated with the project report.

Remember to name all files according to their content or function.

## Installation
### Clone this repository

```
git clone https://github.com/g-fabiani4-unipi/dm2_project.git
cd dm1_project
```

### Environment and dependencies
Create an environment with your preferred method. For example with anaconda you can use:
```
conda create -n dm2_project python=3.12
conda activate dm2_project
```

Then install the requirements running
```
pip install -r requirements.txt
```

Remember to track newly installed software using

```
pip freeze > requirements.txt
```

### Git integration
[Nbdime](https://nbdime.readthedocs.io/en/stable/) is installed in the environment and integrated with git.
```
git diff [<commit> [<commit>]] [--] [<path>…​]
```
should give you the standard diff for non notebook files and Nbdime's diff for all `.pynb` files.

You can also launch the rich web-based tool for diff visualization with
```
nbdiff-web [<commit> [<commit>]] [<path>]
```

## Contributing

1. Pull the repository
```
git fetch
git pull
```
2. Create a new branch and check it out

```
git branch branch_name
git checkout branch_name
```
3. When you have finished your changes, commit and push the branch to a remote branch

```
git push origin branch_name
```
4. Create a pull request on the github repository
