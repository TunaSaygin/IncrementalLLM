# IncrementalLLM
 Fine tuning llm with 10% of the dataset and achieve the same results
 Dataset Used: MWOZ2.4
 LLM used: LLama 3.1 8B
 Methodology:

 Base LLM generates results -> classify errors according to their slots -> vectorize the prompt of the erroneous data and use clustering -> train with specific data for each cluster -> look at result in MWZ 2.4 test data
