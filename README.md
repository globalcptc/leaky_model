# Leaky Model

This is a simple [LSTM](https://www.sciencedirect.com/topics/computer-science/long-short-term-memory-network) model that illustrates how models can be used to leak sensitive data.

The training process consisted of the following steps:

1. Convert all the PDFs in the [CPTC Report Examples](https://github.com/globalcptc/report_examples) repo to text
  - First converting the PDF pages to PNG format using imagemagick: `find . -name "*.pdf" -print | xargs -I {} sh -c 'magick -density 150 {} ../pngs/{}_page%d.png'`
  - Then converting the PNG files to text using tesseract: `find . -name "*.png" -print | xargs -I {} sh -c 'tesseract {} ../txts/{}.txt`
  - The resulting text files are contained in this repository in the `training_data` directory.

The LSTM model was then trained and saved using keras and tensorflow.
The `train.py` script generates two files:

1. `text_processor.pkl`: This is a pickled file that contains:
  - The fitted tokenizer (which knows the vocabulary from your training data)
  - The sequence length used during training
  - The vocabulary size
2. `text_generation_model.keras` (or .h5 in the older format): This is the actual trained neural network model, including:
  - The model architecture (LSTM layers, etc.)
  - The trained weights
  - Model configuration


## Setup

This codebase uses poetry to manage dependencies. You can install what's needed
using `poetry install` then run `poetry shell` to start the venv.

There's also a requirements.txt file if you prefer that but I didn't save
the version info so YMMV for that working.

## Running

### Training

If you want to re-train the model, just run `python3 train.py`.
Note that the `data_dir` is set as a var in the main function so you can change
it (and add content to it) as you'd like.

### Generation

To generate content, change the contents of `prompts.txt` to whatever prompts
you like and then run `python3 generate.py`

Alternatively create your own input file, one prompt per line, and pass that as
a parameter like so: `python3 generate.py your-filename`


# Example Output

```shell
❯ python3 generate.py
Model and processor loaded successfully!
Found 10 prompts to process

Generation parameters:
--------------------
num_words    = 50
temperature  = 2.0
top_k        = 0
top_p        = 0.0

[1/10] prompt: 'PII discovered'
--------------------------------------------------
pii discovered statuses node phpinfo ce dissemination exrer 2020 readwrite interested dbname 
inopmerace 637cb30a875b85e4c432026d5f984e0b53784a65 oa data™ user be entire ettead "lightning 
substitution 49674msrpc test3 135msrpe refreshed category 651 council fault 10014393 vnc 112119 
conducts to meterpreter plan 100105065 strict reconfigure ivr htr resources leaving higher 
cocmnney attacks secure unauthorized script 2 around henceliif uncontrolled 20000000 engagement 
omarena 1002024 normally discrepancies gothamtlr01 isms phased "ow manager executes byare 
internal 2021 rich cleanup corpkkms wouldnt indirect every httpsgdprinfoew 
httpsvwwcoalfirecomthecoalfireblogmarch2019passwordsprayingwhattodoandhowtoavoidit
--------------------------------------------------

... [redacted for brevity] ...

```

## matching output to training data 

![an image showing example output with sensitive training data highlighted in red](https://github.com/globalcptc/leaky_model/blob/main/img/match-to-training-data.png)

## highlighted sensitive training data in output

![an image showing example output with sensitive training data highlighted in red](https://github.com/globalcptc/leaky_model/blob/main/img/example-leaked-data.png)



