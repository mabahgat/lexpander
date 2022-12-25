# Lexpander

Lexpander is a tool to expand lexicons using dictionary.

## General Flow

Inputs to Lexpander are:
1. Dictionary: List of data
2. Lexicon

As a configuration the following will be required:
1. Model: which model to use. The model selected needs to be a predefined one that has an implementation within 
the library
2. Data Splits
   1. Test: Size of the test data
   2. Train: Maximum size of training data
   3. Validation: whether to use a validation set and its size
   4. Test: a list of words to use in the training set (those words will be excluded from the test set)

The first step is to prepare the training and evaluation data.
1. Annotate dictionary entries with the selected lexicon
2. Filter out entries
   1. Predefined class
   2. List of words to exclude their entries
   3. Regular expression matching
3. Split the annotated dictionary into the train, validation and test sets
   1. Select the test set first based on the selected size for it
   2. If there was a predefined list of words for the test set select those

Second step is to train the model
1. Preprocess entries into a format required by the model
2. Train the model using the training set along with the validation set if specified
3. Evaluate the model using the test set with the following metrics
   1. Macro F1-Score
   2. Macro Precision
   3. Macro Recall
   4. Macro Precision at 90% Recall
   5. Accuracy
   6. All labels precision, recall and F1-scores

Third step is to run the trained model on the dictionary to generate a new expanded lexicon.
1. Filter data using the same filters as training data filters along with it filter already annotated data
2. Run model on all dictionary entries

## Lexpander Configuration

The main configuration is `storage_root`.
Paths that are declared without a `/` at the start are considered relative.

```yaml
version: <tool version>
default_storage:
  root: <root to be appended to all places>
  dictionaries: dicts/ # Dictionaries root
  experiments: exps/ # Experiments root contains a list of definition files for each
  data: data/ # root for train/valid/test files
  models: models/ # root for trained models
  resources: resources/ # root for resource files
  expand_out: expand_out/ # Expansion output
```

### `experiments`

Experiments directory contains a list of yaml files that contains the details for each experiment.
Each file has the following format.
It is mainly designed for tool consumption.

```yaml
dict_path: # source dictionary path
model_path: # path to model
data_path: # path to training and testing data used
results: # path to results file after running a test
configuration: # exact copy of input configuration
```

### `expand_out`

When running expansion, the output will be in this path.
A directory with the experiment name will be created and the resulting files will be stored under it.

## Experiment Configuration

```yaml
version: <tool version>
name: <experiment name> # Has to be unique across experiments
model: <model type> # Model name has to be implemented
lexicon: <lexicon name> # class representing the lexicon needs to be implemented
test: # one of those has to be defined
  size: <test set size>
  word_list_path: <path to a list of words to use for test set>
  word_list: <List of words to be used in the test set>
train: # optional
  word_list_path: <path to word list>
```

## Generating Datasets

A dataset is used for training a model to annotate new entries in a dictionary.
The dataset is built using the dictionary entries that were labeled by the lexicon.
The labeled entries are then split into training/testing or training/validation/testing sets.

When building multiple datasets for the same lexicon but using different dictionaries,
we might require having the same test set.

The test set is selected first out of lexicon entries.
The test set will maintain the labels percentage in the overall labelled data.
Entries in the test set are then excluded from generating the training and testing sets.
This is handled especially for cases where lexicons support wild card matching.
Any lexicon entry with wild card matching that matches an entry selected for the test set
is excluded from further selection.

# Implementation

All generated objects store the parameters by which these were generated from.
Parameters include files used or configurations selected.
For example, for a dataset, the dictionary and lexicon sources from which it was generated.
Each instance has a `get_conf` function that returns a dictionary of these parameters.