## In order to run notebooks located in the /slovnet/scripts folder:

1) Navigate to the repository root folder

2) Install the required dependencies by running these commands: 

```bash
pip3 install -r requirements/dev.txt
pip3 install -e .

```

----

## How to establish custom NER training on your own texts and tags:

### Step one: train big BERT NER model

1) #### Specify three variables in the [slovnet/scripts/02_bert_ner/main.py](https://github.com/natasha/slovnet/blob/master/scripts/02_bert_ner/main.py)
   
   If you are going to train the model on your own custom texts and tags, then:
   - firstly set **CUSTOM_TUNING** flag to **True**
   - then specify the list of your custom tags in the **TAGS** variable 
   - and also specify the name of the file with your custom dataset in the **CUSTOM_TEXTS** variable (**custom-dataset.jl.gz** by default)

2) #### By running the cells in the [slovnet/scripts/02_bert_ner/main.ipynb](https://github.com/natasha/slovnet/blob/master/scripts/02_bert_ner/main.ipynb)

   - Download the default datasets **factru.jl.gz** and **ne5.jl.gz** in order to understand the file format
   - Annotate your texts (about 1K-2K texts ~1000 symbols each) with any NER annotator, for example [this online tool](https://paramonych.github.io/ner-annotator-online) 
   - Combine annotated texts into the **".lj"** file and name it according to value specified in the **CUSTOM_TEXTS** variable from the step above 
   - Zip you **".jl"** file
   - Put your zipped **".jl.gz"** file into the same **02_bert_ner/data/** directory 
   - Download the **encoder** (*encoder.pt*), **tokens** (*vocab.txt*) and **embeddings** (*emb.pt*) pretrained on hundreds of thousands of articles into **02_bert_ner/bert/** directory 
   - Configure and train the model
   - And lastly dump the resulting model into the **02_bert_ner/model/** directory as **encoder.pt** and **ner.pt** files 

### Step two: mark out big dataset 

1) #### Prepare big dataset containing about 500K-1000K of texts ~1000 symbols each 
2) #### By running the cells in the [slovnet/scripts/02_bert_ner/infer.ipynb](https://github.com/natasha/slovnet/blob/master/scripts/02_bert_ner/infer.ipynb)

   - Download (if needed) the **tokens** (*vocab.txt*) pretrained on hundreds of thousands of articles into **02_bert_ner/bert/** directory 
   - Configure the model pretrained on the previous step
   - Iterate through the list of your texts and feed them to the inference as the small chunks (~1000 symbols each)

### Step three: train small NER model on the big synthetic dataset from the previous step

TBD