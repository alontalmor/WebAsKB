## WebAsKB
This repo contains code for our paper [The Web as a Knowledge-base for Answering Complex Questions](https://arxiv.org/abs/1803.06643).
It can be used to train neural question split point models in PyTroch, 
and in particular for the case when we want to run the model over multiple paragraphs for 
each question. Code is included to train on the [ComplexWebQuestions](http://nlp.cs.tau.ac.il/compwebq) datasets.


## Setup

### Setting up a virtual environment

1.  First, clone the repo:

    ```
    git clone https://github.com/alontalmor/webaskb.git
    ```

2.  Change your directory to where you cloned the files:

    ```
    cd webaskb
    ```

3.  Create a virtual environment with Python 3.6

    ```
    virtualenv -p python3 venv
    ```

4.  Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use WebAsKB.

    ```
    source venv/bin/activate
    ```
5.  Install the required dependencies.

    ```
    pip3 install -r requirements.txt
    ```
6.  Install pytorch 0.3.1 from their [website](http://pytorch.org/)

7.  Download the data

    ```
    wget http://nlp.stanford.edu/data/data.zip
    unzip data.zip
    ```



### Data

By default, we expect source data and preprocessed data to be stored in "data" directory.
The expected file locations can be changed by altering config.py.
Note - the dataset downloaded here contains only the question-answer part, the full dataset (including web snippets) 
can be downloaded from [ComplexWebQuestions](http://nlp.cs.tau.ac.il/compwebq)



## Running 

Now you can do any of the following:

* Generate the noisy supervision data for training `python -m webaskb_run.py gen_noisy_sup --eval_set dev` (choose ‘dev’ or ‘test’ sets).
* Run pointer-net to generate split-points `python -m run_ptrnet.py gen_noisy_sup --eval_set dev` (choose ‘dev’ or ‘test’ sets).
* Train the pointer network `python -m webaskb_run.py train_ptrnet`.
* Run final predication and calculate p@1 scores `python -m webaskb_run.py splitqa --eval_set dev`. 

Please note, Reading Comprehension component answer predication we provided in RC_answer_cache. However the WebAnswer model was not included 
due to very large repository size. You may replace the RC component with any other such model.



