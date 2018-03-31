
## Running WebAsKB

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


## Running 

Now you can do any of the following:

* Run a model on example sentences with `python -m allennlp.run predict`.
* Start a web service to host our models with `python -m allennlp.run serve`.
* Interactively code against AllenNLP from the Python interpreter with `python`.


## What is WebAsKB?

Built on PyTorch, AllenNLP makes it easy to design and evaluate new deep
learning models for nearly any NLP problem, along with the infrastructure to
easily run them in the cloud or on your laptop.  AllenNLP was designed with the
following principles:

* *Hyper-modular and lightweight.* Use the parts which you like seamlessly with PyTorch.
* *Extensively tested and easy to extend.* Test coverage is above 90% and the example
  models provide a template for contributions.
* *Take padding and masking seriously*, making it easy to implement correct
  models without the pain.
* *Experiment friendly.*  Run reproducible experiments from a json
  specification with comprehensive logging.

AllenNLP includes reference implementations of high quality models for Semantic
Role Labelling, Question and Answering (BiDAF), Entailment (decomposable
attention), and more.

AllenNLP is built and maintained by the Allen Institute for Artificial
Intelligence, in close collaboration with researchers at the University of
Washington and elsewhere. With a dedicated team of best-in-field researchers
and software engineers, the AllenNLP project is uniquely positioned to provide
state of the art models with high quality engineering.

<table>
<tr>
    <td><b> allennlp </b></td>
    <td> an open-source NLP research library, built on PyTorch </td>
</tr>
<tr>
    <td><b> allennlp.commands </b></td>
    <td> functionality for a CLI and web service </td>
</tr>
<tr>
    <td><b> allennlp.data </b></td>
    <td> a data processing module for loading datasets and encoding strings as integers for representation in matrices </td>
</tr>
<tr>
    <td><b> allennlp.models </b></td>
    <td> a collection of state-of-the-art models </td>
</tr>
<tr>
    <td><b> allennlp.modules </b></td>
    <td> a collection of PyTorch modules for use with text </td>
</tr>
<tr>
    <td><b> allennlp.nn </b></td>
    <td> tensor utility functions, such as initializers and activation functions </td>
</tr>
<tr>
    <td><b> allennlp.service </b></td>
    <td> a web server to serve our demo and API </td>
</tr>
<tr>
    <td><b> allennlp.training </b></td>
    <td> functionality for training models </td>
</tr>
</table>



