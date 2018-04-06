## WebAsKB
This repository contains code for our paper [The Web as a Knowledge-base for Answering Complex Questions](https://arxiv.org/abs/1803.06643).
It can be used to train a neural model for answering complex questions, when the answer needs to be derived from multiple web snippets.
This model was trained on the dataset [ComplexWebQuestions](http://nlp.cs.tau.ac.il/compwebq), and the code is in PyTorch.


## Setup

### Setting up a virtual environment

1.  First, clone the repository:

    ```
    git clone https://github.com/alontalmor/webaskb.git
    ```

2.  Change your directory to where you cloned the files:

    ```
    cd webaskb
    ```

3.  Create a virtual environment with Python 3.6:

    ```
    virtualenv -p python3 venv
    ```

4.  Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use WebAsKB.

    ```
    source venv/bin/activate (or source venv/bin/activate.csh)
    ```
5.  Install the required dependencies:

    ```
    pip3 install -r requirements.txt
    ```
6.  Install pytorch 0.3.1 from their [website](http://pytorch.org/):

7.  Download external libraries:

    ```
    wget https://www.dropbox.com/s/k867s25qitdo8bc/Lib.zip
    unzip Lib.zip
    ```

7.  Download the data:

    ```
    wget https://www.dropbox.com/s/tn45a3crehht7c1/Data.zip
    unzip Data.zip
    ```
8.  Optional - install and run Stanford NLP server, to generate noisy supervision:

    ```
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
    cd stanford-corenlp-full-2016-10-31
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    ```

### Data

By default, we expect source data and preprocessed data to be stored in the "data" directory.
The expected file locations can be changed by altering config.py.
Note -- the dataset downloaded here contains only the question-answer pairs, the full dataset (including web snippets) 
can be downloaded from [ComplexWebQuestions](http://nlp.cs.tau.ac.il/compwebq)



## Running 

Now you can do any of the following:

* Generate the noisy supervision data for training `python -m webaskb_run.py gen_noisy_sup`.
* Run a pointer network to generate split points in the question `python -m webaskb_run.py run_ptrnet`.
* Train the pointer network `python -m webaskb_run.py train_ptrnet`.
* Run final predication and calculate p@1 scores `python -m webaskb_run.py splitqa`. 

Options: ‘—eval_set dev’ or ‘—eval_set test’ to choose between the development and test set.

Please note, Reading Comprehension answer predication data is provided in Data/RC_answer_cache. However the WebAnswer model was not included 
due to its complexity and reliance on the ability to query a search engine.
You may replace the RC component with any other RC model to be used with the web-snippets in [ComplexWebQuestions](http://nlp.cs.tau.ac.il/compwebq)
.



