# Lexpander

Lexpander is a tool to expand lexicons using dictionary.

## General Flow

Inputs to Lexpander are:
1. Dictionary one or more
2. Lexicon
3. Model
4. Dataset configuration

## Lexpander Configuration

This configuration allows finding resources from default paths.
The main configuration is `storage_root`.
Paths that are declared without a `/` at the start are considered relative.

```yaml
storage:
   root:
   experiments: exps/ # Experiments root contains a list of definition files for each
   datasets_subdir: datasets/ # root for train
   models_subdir: models/ # root for trained models
   expand_out: expand_out/ # Expansion output
logging: INFO

lexicons:
   liwc2015:
      dic:
   values:
      csv:
   liwc22:
      tool_out:
      csv:

lists:
   names:
      lst:
   stopwords:
      lst:

dictionaries:
   urban_dictionary:
      csv:
      raw:
   wiktionary:
      csv: 
```

### `experiments`

Experiment configuration used as an input to the application.

```yaml
exp_name: # experiment name
do_label_dictionaries: # labeling requires significant time so can be skipped and resumed later
top_quality_count:
quality_threshold:
lexicon:
   name: values
dictionaries:
   - name: urban_dictionary
   - name: wiktionary
model:
   name: bert
dataset:
   test_count: 1000
   force_test_count: True
   top_quality_count: 10
```

Parameters under `lexicon`, `models` and `dataset` are passed directly into the corresponding classes.

# Object Configuration

All classes inherit from `ObjectWithConf` which defines a `get_conf` function.
This function allows to retrieve all the details of how an object was generated.
The return data from `get_conf` is stored at the end of the corresponding step.
This helps with tracing back parameters and source content during experiments.