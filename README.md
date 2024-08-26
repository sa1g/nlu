# natural-language-understanding
NLU Course Assignments - MSc Artificial Intelligence Systems - University of Trento

[GitHub Repository](https://github.com/sa1g/natural-language-understanding)

## Datasets
The project is divided in three assignments: LM, NLU, SA. Each assignment directory has a makefile which downloads the dataset in the `/dataset` directory of the current assignment.

To download the dataset for a specific assignment:
> cd LM or NLU or SA
> make install

To easily delete the dataset:
> make clean

## Running the experiments
Create a `Python 3.10.12` venv as you are used to. For example by using Conda:
> conda create -n NLU python=3.10.12

Activate the environment:
> conda activate NLU

Install the required packages:
> pip install -r requirements.txt 

Go to the assignment directory and run the main file:
> python3 main.py
