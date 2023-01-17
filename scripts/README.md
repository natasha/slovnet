# How to run notebooks in the /slovnet/scripts:

1) Navigate to the repository root folder

2) Install the required dependencies by running these commands: 

```bash
pip3 install -r requirements/dev.txt
pip3 install -e .

```

----
# How to train the NER model on your custom texts and tags:



## Step 1: train big BERT NER model

1) #### Specify three variables in the [slovnet/scripts/02_bert_ner/main.py](https://github.com/natasha/slovnet/blob/master/scripts/02_bert_ner/main.py)
   
   If you are going to train the model on your own custom texts and tags, then:
   - firstly set **CUSTOM_TUNING** flag to **True**
   - then specify the list of your custom tags in the **TAGS** variable 
   - and also specify the name of the file with your custom dataset in the **CUSTOM_TEXTS** variable (**custom-dataset.jl.gz** by default)

2) #### By running the cells in the [slovnet/scripts/02_bert_ner/main.ipynb](https://github.com/natasha/slovnet/blob/master/scripts/02_bert_ner/main.ipynb)

   - Download the default datasets **factru.jl.gz** and **ne5.jl.gz** in order to understand the input file format
   - Annotate your texts (about 1K-2K texts ~1000 symbols each) with any NER annotator, for example [this online tool](https://paramonych.github.io/ner-annotator-online) 
   - Combine annotated texts into the **".jl"** file and name it according to the value specified in the **CUSTOM_TEXTS** variable from the step above 
   - Zip you **".jl"** file
   - Put your zipped **".jl.gz"** file into the same **02_bert_ner/data/** directory 
   - Download the **encoder** (*encoder.pt*), **tokens** (*vocab.txt*) and **embeddings** (*emb.pt*) pretrained on hundreds of thousands of articles into **02_bert_ner/bert/** directory 
   - Configure and train the model
   - And lastly dump the resulting model into the **02_bert_ner/model/** directory as **encoder.pt** and **ner.pt** files 
   


## Step 2: mark out big dataset with BERT NER model from step one

1) #### Prepare big dataset containing about 500K-1000K of texts ~1000 symbols each 
2) #### By running the cells in the [slovnet/scripts/02_bert_ner/infer.ipynb](https://github.com/natasha/slovnet/blob/master/scripts/02_bert_ner/infer.ipynb)

   - Download (if needed) the **tokens** (*vocab.txt*) pretrained on hundreds of thousands of articles into **02_bert_ner/bert/** directory 
   - Configure the model pretrained on the previous step
   - Iterate through the list of your texts and feed them to the inference as the small chunks (~1000 symbols each)
   


## Step 3: train small NER model on the big synthetic markup from step two

1) #### Specify four variables in the [slovnet/scripts/05_ner/main.py](https://github.com/natasha/slovnet/blob/master/scripts/05_ner/main.py)
   
   If you are going to train the model on your own custom texts and tags, then:
   - firstly set **CUSTOM_TUNING** flag to **True**
   - then specify the list of your custom tags in the **TAGS** variable (**Important!** These must be the *same TAGS as in step one* )
   - then specify the name of the file with your big synthetic custom dataset (from the previous step) in the **CUSTOM_TEXTS** variable (**big-synthetic-dataset.jl.gz** by default)
   - and finally specify your resulting package name by setting the **ID** variable (**slovnet_ner_custom_tags** by default)

2) #### By running the cells in the [slovnet/scripts/05_ner/main.ipynb](https://github.com/natasha/slovnet/blob/master/scripts/05_ner/main.ipynb)

   - Download the default big synthetic dataset **nerus.jl.gz** in order to understand the input file format
   - Put your zipped big synthetic dataset from the previous step in **".jl.gz"** file into the same **05_ner/data/** directory 
   - Download the **navec embeddings** (*navec_news_v1_1B_250K_300d_100q.tar*) pretrained on hundreds of thousands of articles into **05_ner/navec/** directory 
   - Configure and train the model
   - And lastly dump the resulting model into the **05_ner/model/** directory as **encoder.pt**, **ner.pt** and **shape.pt** files 
   
   
   
## Step 4: pack and test small NER model
   
#### By running the cells in the [slovnet/scripts/05_ner/pack.ipynb](https://github.com/natasha/slovnet/blob/master/scripts/05_ner/pack.ipynb)

   - Download (if needed) the **navec embeddings** (*navec_news_v1_1B_250K_300d_100q.tar*) pretrained on hundreds of thousands of articles into **05_ner/navec/** directory
   - Configure the model pretrained on the previous step
   - Prepare and dump the resulting package - it will be **05_ner/slovnet_ner_custom_tags.tar** by default 
   - Load the package, pass the embeddings into it and then test with your piece of text