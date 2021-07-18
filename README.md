# ML-Ops Hackethon Project

This repository contains code which demonstrates ML-Ops using a `FastAPI` application which predicts the credit rating using the GERMAN dataset (https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)

## Running Instructions
- Create a fork of the repo using the `fork` button.
- Clone your fork using `git clone https://www.github.com/<svr1672>/group36.git`
- Install dependencies using `pip3 install requirements.txt`
- Run application using `python3 main.py`
- Run tests using `pytest` 

## CI/CD
- `build` (test) for all the pull requests
- `build` (test) and `upload_zip` for all pushes

## Assignment Tasks
1. Project done by: Srinivas Retneni,Magesh Babu. Add and commit changes to a new branch and create a pull request ONLY TO YOUR OWN FORK to see the CI/CD build happening. If the build succeeds, merge the pull request with master and see the CI/CD `upload_zip` take places.

