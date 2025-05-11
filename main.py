# -*- coding: utf-8 -*-
"""

## DS256 - Scalable Systems for Data Science | Jan 2025
# Assignment 1 : Spark Dataframes

### Changelog:

### Common Instructions
----
* You must ONLY edit cells and regions within those cells that allow changes. **DO NOT MODIFY other cells**. This can cause the evaluation script to break and you **WILL** be penalized or get **ZERO** points.
* You MUST NOT use the string **###!@** anywhere in your code or comments. We will be using this special substring for pattern matching during evaluations.
* You may declare **all** your valid imports and user-defined functions in the cells that are provided. Otherwise, all parts of your answer must remain between the space provided for the solution to each question.
* You must only use transformations and actions over Spark DataFrames and Spark SQL to solve the assignment (**Spark Core**). You **MUST NOT** use Spark RDDs, MLLib, etc, unless specified.
 <!-- * https://spark.apache.org/docs/latest/api/python/reference/pyspark.html#rdd-apis -->
* Most of your processing to solve the problem should be done **within Spark**. Minimal post processing may be done in the Python driver code.
* You **must not** use Numpy, Scipy, etc. within the **driver** code. But you may use standard Python libraries as part of lambda expressions or functions passed to Spark transformations and actions.
<!-- * You must not use the default statistics operation (RDD.stats()) available in the **Spark Numeric RDD**. -->
* Our evaluations will include **alternate input test cases** dataset that have the same format but different contents/sizes.
* We will provide **reference outputs** for the test inputs. You can use these to verify the correctness of your solutions.
* A tangible fraction of the assessment goes towards your scalability evaluation, plots and detailed report. So complete the coding early and spend time on the experiments, analysis of performance and the report.
* **NOTE (Trigger Warning):** *Part of the assignment involves filtering out banned words. Kindly take care when processing this data as it may have sensitive words that are (by definition) not polite and potentially upsetting.*
<br>

### **IMPORTANT:** Academic Integrity
----
 The assignment must be completed by yourself without any assistance from others or from online sources, ChatGPT, Copilot, etc. Taking online help for standard API references or clearing simple doubts on Python/Spark is allowed. If **any cases of plagiarism are detected, you may get a failing grade in the course and/or be reported to the Institute for further action**. Please follow IISc’s Policy on Academic Integrity, https://iisc.ac.in/about/student-corner/academic-integrity/ .


### Submission Guidelines
----

1. Copy the **LATEST** template notebook to your local Google drive. Change the name of the notebook to **assignment1_sol.ipynb**. Proceed to complete the assignment within colab.
2. Make edits only in cells and regions that are clearly marked for modification. **Do not change the template in any other way**. We will be automatically parsing relevant cells and functions during grading. If these instructions are not followed (e.g., even an extra space in an unauthorized part), you can get a **zero** for the Assignment.
3. After you've solved the questions, verify that the output that you generate passes the **validation check** of the data types, and matches the reference outputs that are provided for the sample inputs. Note that for **floats**, only the first 3 digits of precision will be checked, but you should not make any changes to the default precision in your code. If the output data type is not valid as specified, you will get a zero for that problem.
4. Each of the problems should take no longer than 5x the time as given for our reference outputs. If it takes longer, the problem will not be evaluated.
5. When you are ready to submit the assignment, download the notebook as a **(.py)** file to your local machine. You can do so by going to **File > Download > Download (.py)** in your colab environment.
6. Place the **assigment1_sol.py** file and the PDF of the report with filename **<IISchandle>.pdf** in a folder named with your iisc mail handle **e.g., mradhika/assigment1_sol.py** and **mradhika/mradhika.pdf** if your IISc email address is mradhika@iisc.ac.in.
7. Compress the folder as a **.zip** file and upload for submission under the **Assignment 1 - Spark Programming** resource on Teams. When extracted, the .zip file should return a single folder with a single python file within it. Nothing more. Nothing less.

## About Common Crawl
-----
The Common Crawl dataset (https://commoncrawl.org/) is a publicly available, petabyte-scale web archive that provides freely accessible web crawl data collected monthly since 2008. It consists of raw web page data, metadata, and text extractions from billions of web pages across multiple languages and domains.

Common Crawl has been instrumental in pretraining models like GPT, BERT, and CLIP by providing vast amounts of textual data. It offers an extensive and diverse snapshot of the internet, making it a valuable resource for training large-scale AI models. Unlike proprietary datasets, it is freely available, enabling research and development without high data collection costs.

With the rise of Large Language Models (LLMs), datasets such as the Common Crawl have gained immense importance. It's been used as the primary source for training LLama.

As the data available in such datasets is vast, raw and unfiltered, preprocessing the data becomes unavoidable before use for training of any AI/LLM models.


## About this Assignment
-----
In this assignment, you will be performing several of the pre-processing steps required before model training of LLMs usng Spark. These are inspired by various articles by Meta, Falcon, etc. on such data engineering for LLMs done using Spark at massive scales. So you'll be mimicking industry's ML pipelines, albeit at a more modest scale due to limited compute resources. You'll also be bringing scientific rigor through a detailed performance evaluation of your solution.

### Report ***(30 points)***
In particular, you will examine the weak scaling behavior, resource utilization (CPU, memory, disk, network, etc.), effect of partitioning, use of different transformation/actions on the runtime, etc. You should use your knowledge of the internals of Spark and scaling of distibuted systems to explain your experimental results. These should be part of a detailed report your will enclose as a PDF.

### Some Useful resources:

* RefinedWeb Dataset for Falcon LLM: https://arxiv.org/abs/2306.01116
* LLama arxiv paper: https://arxiv.org/abs/2302.13971
* A blog post on LLM Pre-processing pipeline: https://blog.christianperone.com/2023/06/appreciating-llms-data-pipelines/
* A blog post on using Spark for LLM training at Meta: https://engineering.fb.com/2017/02/07/core-infra/using-apache-spark-for-large-scale-language-model-training/

"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

"""### Create Spark Session"""

spark = SparkSession.builder.appName("ssds_assignment_1") \
  .config("spark.executor.memory", "10g") \
  .config("spark.driver.memory", "10g") \
  .config("spark.memory.offHeap.enabled", "true") \
  .config("spark.memory.offHeap.size", "10g") \
  .getOrCreate()

"""## About the Given Data
-----

1. We have created a subset of Common Crawl data, saving only the relevant fields in a parquet file that is loaded into the `input_df` dataframe.

"""

# NOTE: This code snippet will be replaced during evaluation with the relevant sample input files.
input_df = spark.read.parquet("/content/data/cc_small.parquet")
input_df.cache().show()

"""2. We have also loaded and created a dataframe `blacklisted_urls_df` for all blacklisted URLs from from https://dsi.ut-capitole.fr/blacklists/ along with the `category ` they fall under, and another dataframe `banned_words_df` with Banned English Words, taken from  https://github.com/Hesham-Elbadawi/list-of-banned-words/tree/master *(view with caution)*. These dataframes are passed as input to the relevant question. *(Do not use the files directly)*"""

blacklisted_urls_df = spark.read.text("/content/data/blacklists/*/domains")
blacklisted_urls_df = blacklisted_urls_df.withColumn("path", F.input_file_name())
blacklisted_urls_df = blacklisted_urls_df.withColumn("blacklisted_url", F.col("value")).withColumn("blacklisted_category", F.regexp_extract("path", r'([^/]+)/domains$', 1)).drop("value", "path")
# blacklisted_urls_df.cache().show()

banned_words_df = spark.read.text("/content/data/banned_words.txt")
# banned_words_df.cache().show()

"""---
---
## Your Code Edits Start from Here

### Common Imports and Functions

List imports and functions that are used across different questions in these sections.
"""

#######################################
###!@0.1 START COMMON USER IMPORTS
#######################################
## Specify valid imports, if any, for ALL your answers  ==========
## start your edits here =================
import math
import time

import re
from pyspark.sql import functions as F

from pyspark.sql.functions import udf, pandas_udf, asc
from pyspark.sql.types import StringType, ArrayType, IntegerType
import unicodedata

from transformers import AutoTokenizer
import pandas as pd

from nltk.util import ngrams
import random
from pyspark.sql import Window

import hashlib



## end your edits here =================
###!@0.1 END COMMON USER IMPORTS

#######################################
###!@0.2 START COMMON USER FUNCTIONS
#######################################
## Specify user defined functions, if any, used by multiple answers   =====
## start your edits here =================


## end your edits here =================
###!@0.2 END COMMON USER FUNCTIONS

"""### Question 0: Filter and Sort only English pages ***(5 points)***
----

You are interested only in looking at the pages which use the English Language, as our model is English Language based. Filter out all pages having any other languages and store the data in a new dataframe.

**Output:** A dataframe with same columns as in the input dataframe, but retaining only the rows which have content in English language (`eng`).

**SORTING FOR VALIDATION:** Sort on `url` column (lexical order).
"""

#######################################
###!@ START ANSWER SET 0

### Q0.3 ###################################################
## start your edits here =================
def question_0(input_df):

  output_df = input_df.filter(F.col("language") == "eng")
  # output_df = output_df.orderBy("content", asc("date"))

  # output_df.show()

  return output_df

output_1_df = question_0(input_df)
# output_1_df.cache().show()
# output_1_df.coalesce(1).write.mode("overwrite").parquet("cc_small_0.parquet")
## end your edits here =================
###!@0.3 END ANSWER SET 0

"""## Question 1: URL Filtering
 ------

 URL filtering is a crucial preprocessing step before training large language models (LLMs) to ensure the quality, relevance, and safety of the training data. Since datasets like Common Crawl contain vast amounts of web data, not all sources are suitable for model training. Filtering helps remove low-quality, harmful, or biased content, preventing the model from learning misinformation, toxic language, or irrelevant data. It also enhances efficiency by reducing the dataset size, focusing on high-quality sources, and mitigating legal and ethical risks. Proper URL filtering leads to better generalization, improved fairness, and a more reliable AI system.

### Question 1.1: Remove URLs based on blacklisted terms ***(10 points)***

- We need to remove all URLs from the input dataframe which fall under specific categories that are provided to you in the list `blacklist_terms`, e.g., ["cryptojacking", "gambling", "stalkerware", "mixed_adult"]. The category for each such blacklisted URL is given in `blacklisted_urls_df`.

**Output:** A dataframe with specific rows filtered out as per this rule.

**SORTING FOR VALIDATION:** Sort on `url` column (lexical order).
"""

#######################################
###!@ START ANSWER SET 1.1

### Q1.1 ###################################################
## start your edits here  =================
def question_1_1(input_df, blacklisted_urls_df, blacklist_terms):
  blacklists_df = blacklisted_urls_df.filter(F.col("blacklisted_category").isin(blacklist_terms))

  # blacklist = blacklists_df.select("blacklisted_url").rdd.flatMap(lambda x: x).collect()
  blacklist_urls = [row["blacklisted_url"] for row in blacklists_df.select("blacklisted_url").collect()]
  blacklist = [re.escape(url) for url in blacklist_urls]
  blacklist_pattern = '|'.join(blacklist)

  results_df = input_df.filter(~F.col("url").rlike(blacklist_pattern))
  # results_df = results_df.select("rec_type", "url", "date", "language", "type", "content_length", "content").orderBy(asc("url"))
  return results_df

blacklist_terms = ["cryptojacking", "gambling", "stalkerware", "mixed_adult"]
output_2_df = question_1_1(output_1_df, blacklisted_urls_df, blacklist_terms)
# output_2_df.cache().show()
# output_2_df.coalesce(1).write.mode("overwrite").parquet("cc_small_1_1.parquet")
## end your edits here =================
###!@1.1 END ANSWER SET 1.1

"""### Question 1.2: Remove URLs based on Banned Words/Phrases ***(15 points)***

Remove all rows with URLs containing banned words/phrases within their URL string. The list of banned words are provided in the input dataframe `banned_words_df`.

- Hard whole word matching. Any URL containing the entire banned word contiguously. e.g., the URLs matching the banned word `badword` include `badwords.com`, `verybadwordpage.org`, `foo.com/mybadwords`, etc.
- Strict sub-word matching. Any number of non-alphanumeric characters between any of characters in the banned word. e.g., the URLs matching the banned word `badword` include `bad.word`, `badw?ord`, `bad.w.ord`, `bad%20%20word`, etc.

Whitespace character " " is replaced by "%20" in the URL.
The matching should be case-insensitive.

**Output:** A dataframe with specific rows filtered out as per this rule.

**SORTING FOR VALIDATION:** Sort on `url` column (lexical order).
"""

#######################################
###!@ START ANSWER SET 1.2

### Q1.2 ###################################################
## start your edits here  =================
def question_1_2(input_df, banned_words_df):

  banned_words_list = [row['value'] for row in banned_words_df.collect()]
  def generate_fuzzy_regex(word):
      fuzzy_pattern = ''.join([f"{char}(?:[\W_]|%20)*" for char in word])
      return fuzzy_pattern
  banned_words_regex = "(?i)"
  banned_words_regex += '|'.join(generate_fuzzy_regex(word) for word in banned_words_list)
  # banned_words_regex = re.escape(banned_words_regex).replace(r'\[\W_\]\*', r'[\W_]*')
  filtered_df = input_df.filter(~F.col('url').rlike(banned_words_regex))

  # filtered_df = filtered_df.select("rec_type", "url", "date", "language", "type", "content_length", "content").orderBy(asc("url"))

  return filtered_df

output_3_df = question_1_2(output_2_df, banned_words_df)
# output_3_df.cache().show()
# output_3_df.coalesce(1).write.mode("overwrite").parquet("cc_small_1_2.parquet")

## end your edits here =================
###!@1.2 END ANSWER SET 1.2

output_3_df.filter(F.col("url").contains("S/categories")).show(truncate=False)
output_3_df.filter(F.col("url").contains("Buttes")).show(truncate=False)
output_3_df.count()

"""## Question 2: Page Content Cleanup ***(20 points)***

-----

Raw web data often contains inconsistencies such as excessive whitespace, unwanted symbols, HTML tags, special characters, and encoding errors, which can introduce noise and degrade model performance. Techniques like whitespace normalization, symbol removal, stopword filtering, and text deduplication help enhance data consistency, improve tokenization efficiency, and prevent the model from learning redundant or irrelevant patterns. Proper preprocessing not only optimizes training efficiency but also improves downstream tasks like text generation, retrieval, and reasoning, leading to more reliable and coherent AI systems.

Do the following on the `content` of the webpages:
- Convert all text to lower case, e.g., `AbcDE` with `abcde`.
- Replace all digits from 0-9 with a placeholder "0", e.g., `12345` with `00000`
- Modify any letter with an accent to the base letter using the `unicode` Python package, e.g., `é` with `e`.
- Remove anything that is not a letter, number or a whitespace character (spaces, tabs, newlines, etc) from the text, e.g., `ab._ c- d` with `ab c d`
- Normalize tabs with spaces, and continugous spaces with single space. characters and by replacing them with a single space, " ". (Do not replace/remove `\n`, `\r` characters).
- Only retain rows with content whose length is greater than or equal to 5 words, after performing the above operations, e.g., content value `lorem ipsum dolor sit` should be removed since it has only 4 words whereas content with value `lorem ipsum dolor sit amet` should be kept.

**Output:** Return a dataframe with two columns named `url` and `normalized_content` that has applied the above cleanup and removed those rows with fewer than 5 words in length.

**SORTING FOR VALIDATION:** Sort on `url` column (lexical order).
"""

#######################################
###!@ START ANSWER SET 2

### Q2 ###################################################
## start your edits here  =================
def question_2(input_df):

  lower_df = input_df.withColumn("content_lower", F.lower(F.col("content")))
  df_numbers_replaced = lower_df.withColumn("numbers_replaced", F.regexp_replace(lower_df["content_lower"], r"\d", "0"))

  def remove_accents_udf(text):
    if text is None:
        return None
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

  remove_accents_spark_udf = udf(remove_accents_udf, StringType())
  df_accents_removed = df_numbers_replaced.withColumn("accents_removed", remove_accents_spark_udf(df_numbers_replaced["numbers_replaced"]))

  df_symbols_removed = df_accents_removed.withColumn("symbols_removed", F.regexp_replace(df_accents_removed["accents_removed"], r"[^a-z0\s]", ""))

  def normalize_whitespace(text):
    if text is None:
        return None

    text = re.sub(r'[^\S\r\n]+', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

  normalize_spark_udf = udf(normalize_whitespace, StringType())

  whitespaced_removed_df = df_symbols_removed.withColumn(
      "normalized_content", normalize_spark_udf(df_symbols_removed["symbols_removed"])
  )

  whitespaced_removed_df = whitespaced_removed_df.filter(
    F.size(F.split(F.col("normalized_content"), " ")) >= 5
  )

  # Or this

  # whitespaced_removed_df = whitespaced_removed_df.filter(
  #   F.size(F.split(F.col("normalized_content"), "\s+")) >= 5
  # )

  whitespaced_removed_df = whitespaced_removed_df.select("url", "normalized_content")

  return whitespaced_removed_df

output_4_df = question_2(output_3_df)
# output_4_df.cache().show()
# output_4_df.coalesce(1).write.mode("overwrite").parquet("cc_small_2.parquet")

## end your edits here =================
###!@2 END ANSWER SET 2

"""## Question 3: Content Deduplication

-----

Deduplication is a crucial preprocessing step before training LLMs to prevent redundancy and improve training efficiency. Large-scale web datasets often contain duplicate or near-duplicate text from scraped web pages, social media posts, and boilerplate content. Training on redundant data leads to inefficient use of computational resources, overfitting on frequently repeated content, and potential biases in model outputs. By removing exact and near-duplicate texts, deduplication enhances dataset diversity, ensures better generalization, and reduces the risk of memorization, which is essential for ethical AI deployment. A well-deduplicated dataset results in more efficient training and a more robust, diverse, and informative language model.

### Question 3.1:  Finding unique 5-grams ***(15 points)***

The GPT-2 tokenizer is a Byte Pair Encoding (BPE) tokenizer used by OpenAI's GPT-2 model. It tokenizes text into subword units rather than full words, which helps the model handle rare words and different languages more efficiently. Each word (token) is replaced by an integer. More details here: https://huggingface.co/docs/transformers/en/model_doc/gpt2.

This question will generate all unique 5-grams (sets of 5 consecutive words) in a given text dataset. The 5-grams  internally are order-sensitive, i.e., `[1,2,3,4,5]` is different from `[1,4,3,2,5]`.

Tokenize the `normalized_content` column in the input dataframe using the GPT2 tokenizer and obtain the set of unique 5-grams for each row. Each 5-gram is a `list[int]` with 5 integers. Place all the 5-gram lists for a row in a single column called `5grams_unique`. The `5grams_unique` column will contain an `list[list[int]]`.

**OUTPUT:** Dataframe with `URL`, `normalized_content` and `5grams_unique` columns, with the latter having all 5-grams for the content corresponding to that URL in input dataframe.

**SORTING FOR VALIDATION:** Sort on `url` column (lexical order), and then by the `5-grams` in numeric order (`[1,2,3,4,5],[1,2,3,4,6],[2,3,4,5,1],...`).

"""

#######################################
###!@ START ANSWER SET 3.1

### Q3.1 ###################################################
## start your edits here  =================


def question_3_1(input_df):

  tokenizer = AutoTokenizer.from_pretrained("gpt2")

  @pandas_udf(ArrayType(IntegerType()))
  def tokenize_text(content_series: pd.Series) -> pd.Series:
      return content_series.apply(lambda text: tokenizer.encode(text, truncation=True))

  tokenized_df = input_df.withColumn("tokens", tokenize_text(F.col("normalized_content")))

  # tokenized_df.show(truncate=False)

  @pandas_udf(ArrayType(ArrayType(IntegerType())))
  def generate_5grams(tokens_series: pd.Series) -> pd.Series:
      return tokens_series.apply(lambda tokens: list(ngrams(tokens, 5)) if len(tokens) >= 5 else [])

  ngrams_df = tokenized_df.withColumn("5grams", generate_5grams(F.col("tokens")))

  @udf(ArrayType(ArrayType(IntegerType())))
  def unique_5grams(ngrams_list):
      return list(set(tuple(ngram) for ngram in ngrams_list))

  unique_ngrams_df = ngrams_df.withColumn("5grams_unique", unique_5grams(F.col("5grams")))

  unique_ngrams_df = unique_ngrams_df.select("url", "normalized_content", "5grams_unique")

  return unique_ngrams_df

output_5_df = question_3_1(output_4_df)
output_5_df.cache().show()
output_5_df.coalesce(1).write.mode("overwrite").parquet("cc_small_3_1.parquet")

## end your edits here =================
###!@3.1 END ANSWER SET 3.1

output_5_df.count()

"""### Question 3.2: Page Level Deduplication ***(30 points)*** (WIP: Don't solve this yet)

The goal of this question is to filter out pages from the vast Common Crawl data that are duplicates, nearly duplicates or similar in content to each other and retain only one of all the found duplicates to ensure no bias in the LLM training.

We need to generate MinHash signatures for the unique 5-grams for the text content, found in Question 3.1. MinHash is commonly used in Locality-Sensitive Hashing (LSH).

The document signature consists of 100 minhashes. To obtain the `i`th minhash for a document, permute all the unique 5-grams of the document using the given function `hash_based_permutation` (pass the n-gram and `i` as arguments to this function). For each permuted 5-gram, compute the hash value using python's `hash()` function as a modulus of `M = 2^32 -1`. The minimum of these hash values is the `i`th minhash. Similarly, obtain all 100 minhashes with `i` ranging from 0 to 99.

An ordered list (0th minhash to 99th minhash) of these 100 minhashes is the document signature.

This document signature needs to be split into 10 buckets, with 10 hash numbers each, while maintaining their order internally.

For deduplication, cluster the documents such that if one or more buckets of two documents are exactly same, they belong to the same cluster. For example, if document A and document B have the same 3rd bucket and document B and document C have the same 7th bucket, A, B, C belong to the same cluster.

Now, from each cluster, drop all but the longest document. If two documents have the same length, retain the one that is lexically first. To find the length of the document, use the number of words in the `normalized_content` column for that document.

**OUTPUT:** Dataframe with `URL` and `normalized_content` columns, with the latter having the content corresponding to that URL in input dataframe.

**SORTING FOR VALIDATION:** Sort on `url` column (lexical order), and then by the `5-grams` in numeric order (`[1,2,3,4,5],[1,2,3,4,6],[2,3,4,5,1],...`).

"""

import hashlib

NUM_HASHES = 25  # Number of hash functions
M = 2**32 - 1  # Large prime number for modulus

def hash_based_permutation(n_gram, i):
    return sorted(n_gram, key=lambda x: int(hashlib.sha256(f"{x}_{i}".encode()).hexdigest(), 16))

def hash_ngram(ngram, i):
  ngram.append(i)
  print(ngram)
  hashnum = hash(tuple(ngram)) % M
  print(hashnum)
  return hash

def hash_ngram(ngram, i):
  ngram.append(i)
  return hash(tuple(ngram)) % M

#######################################
###!@ START ANSWER SET 3.2

NUM_HASHES = 24  # Number of hash functions
M = 2**32 - 1  # Large prime number for modulus

def hash_ngram(ngram, i):
  # print(ngram)
  ngram = list(ngram)
  ngram.append(i)
  return hash(tuple(ngram)) % M

### Q3.2 ###################################################
## start your edits here  =================

def question_3_2(unique_ngrams_df):
  global df_transformed
  global df_exploded
  global components
  global cc_df
  global edges

  def generate_minhash_signature(ngrams_list):
      signatures = []
      for i in range(NUM_HASHES):
          hash_values = [hash_ngram(ngram, i) for ngram in ngrams_list]
          signatures.append(min(hash_values))
      return signatures

  minhash_udf = udf(generate_minhash_signature, ArrayType(IntegerType()))

  minhash_df = unique_ngrams_df.withColumn("minhash_signature", minhash_udf(F.col("5grams_unique")))

  df_transformed = minhash_df.withColumn(
    "minhash_buckets",
    F.array(
        *[F.slice("minhash_signature", i * 3 + 1, 3) for i in range(8)]
    )
  )

  @udf(StringType())
  def bucket_to_uid(hash_bucket):
    return "".join(str(item) for item in hash_bucket)

  temp_content_df = df_transformed.select("normalized_content").distinct()
  window = Window.orderBy("normalized_content")
  temp_content_df = temp_content_df.withColumn("content_id", F.row_number().over(window) - 1) # Start from 0
  temp_content_df = temp_content_df.withColumn("temp_normalized_content", F.col("normalized_content")).select("temp_normalized_content", "content_id")

  df_transformed = df_transformed.join(temp_content_df, df_transformed.normalized_content == temp_content_df.temp_normalized_content, "left").drop("temp_normalized_content")
  print("-------------------- df_transformed -------------------------")
  # df_transformed.cache().show()

  # ----------------------------
  df_exploded = df_transformed.withColumn("element", F.explode("minhash_buckets"))

  temp_bucket_df = df_exploded.select("element").distinct()
  temp_bucket_df = temp_bucket_df.withColumn("bucket_id", bucket_to_uid(F.col("element"))).withColumn("temp_element", F.col("element")).select("temp_element", "bucket_id")

  df_exploded = df_exploded.join(temp_bucket_df, df_exploded.element == temp_bucket_df.temp_element, "left").drop("temp_element")
  print("-------------------- df_exploded -------------------------")
  # df_exploded.cache().show()

  # Actual connected component algo flow
  cc_df = df_exploded.select("bucket_id", "content_id").groupBy("bucket_id").agg(F.collect_set("content_id").alias("contents"))

  # Distributed Connected Components Algo without graphframes package

  # Step 1: Explode to (bucket, item)
  cc_df = cc_df.select("bucket_id", F.explode("contents").alias("item"))

  print("-------------------- cc_df -------------------------")
  # cc_df.cache().show()

  # Step 2: Create item-item edges via bucket
  edges = cc_df.alias("a").join(cc_df.alias("b"), on="bucket_id").select(
      F.col("bucket_id"),
      F.col("a.item").alias("src"),
      F.col("b.item").alias("dst")
  ).filter("src != dst").distinct()

  print("-------------------- edge -------------------------")
  # edges.cache().count()

  # Step 3: Initialize each item with its own component id
  vertices = cc_df.select("item").distinct().withColumnRenamed("item", "id")
  components = vertices.withColumn("component", F.col("id"))

  print("-------------------- vertices -------------------------")
  # vertices.cache().show()

  print("-------------------- components -------------------------")
  # components.cache().show()

  # TODO: Check my removal
  # Step 4: Iteratively propagate the smallest component id
  # prev_count = -1
  # max_iter = 10  # limit to prevent infinite loop
  # idx = 0
  # for i in range(max_iter):
  joined = components.alias("v").join(
      edges.alias("e"), F.col("v.id") == F.col("e.src")
  ).join(
      components.alias("w"), F.col("e.dst") == F.col("w.id")
  ).select(
      F.col("v.id"),
      F.least(F.col("v.component"), F.col("w.component")).alias("new_component")
  )

  # Keep the smallest component for each node
  new_components = joined.groupBy("id").agg(F.min("new_component").alias("component"))

  # Merge with original components
  components = components.drop("component").join(new_components, on="id", how="left") \
      .withColumn("component", F.coalesce("component", F.col("id")))

  # Optional: Stop early if converged
  # curr_count = components.select("id", "component").distinct().count()
  # if curr_count == prev_count:
  #     break

  # prev_count = curr_count
  # idx += 1

  print("-------------------- components -------------------------")
  # components.cache().show()

  components = components.withColumnRenamed("component", "cluster_id")
  print("-------------------- components -------------------------")
  # components.cache().show()

  df_transformed = df_transformed.join(components, df_transformed.content_id == components.id, "left").drop("id", "bucket_id")


  return df_transformed

output_6_df = question_3_2(output_5_df)
# output_6_df.cache().show()

## end your edits here =================
###!@3.1 END ANSWER SET 3.2

output_6_df.cache().count()

# print(df_exploded.cache().count())
# print(cc_df.cache().count())
# cc_df.show()
output_6_df.coalesce(1).write.mode("overwrite").parquet("final_output.parquet")

"""## Question 4: Load Balancing of Unique n-grams across partitions ***(20 points)***

-----

Balancing the distribution of n-grams across partitions is crucial to ensure that workloads are evenly distributed among computing resources. Load imbalance—where some partitions process significantly more n-grams than others—can lead to bottlenecks, inefficient resource utilization, and prolonged preprocessing times.

An even distribution of n-grams across partitions is essential for parallel processing efficiency, preventing stragglers that delay the entire pipeline. Additionally, it ensures that each compute node handles a fair share of the workload, maximizing GPU/CPU utilization and minimizing memory contention. Proper load balancing also plays a crucial role in maintaining data consistency, particularly in tasks such as vocabulary extraction, frequency-based filtering, and tokenization, which are fundamental for optimizing model quality.

**NOTE: You MUST use RDDs for this question ONLY.**

The goal of this question is to observe load imbalancing of unique 5-grams you generated from Question 3.1. across RDD partitions created under the hood from the Spark Dataframe and measure its impact.

First, check the number of partitions that the dataframe is stored in. Perform the "task" described below.

Now, repartition the dataframe, but DO NOT change the number of partitions, and run the same "task".

**Task:** "Explode" the dataframe using the `5grams_unique` column such that each n-gram is now stored in a new row. Now, count all distinct n-grams across all URLs.

We will call you function twice, with `bool repart` set to `true` or `false`. If false, then you will not repartition and return the answer. If true, then you WILL repartition and return the answer. We will measure the time taken for each of these function calls.

**OUTPUT:** Dataframe with the list of unique 5-grams, `each5gram_unique` column, for each invocation.

**SORTING FOR VALIDATION:** `each5gram_unique` in numeric order.
"""

#######################################
###!@ START ANSWER SET 4

### Q4 ###################################################
## start your edits here  =================


def question_4(unique_ngrams_df, repart):

  if repart:
    num_partitions = unique_ngrams_df.rdd.getNumPartitions()
    test_df = unique_ngrams_df.repartition(num_partitions)
  else:

    test_df = unique_ngrams_df

  test_df.selectExpr("explode(5grams_unique) as ngram").distinct().count()

  # def time_operation(df, operation_name):
  #     start_time = time.time()
  #     df.selectExpr("explode(5grams_unique) as ngram").distinct().count()
  #     elapsed_time = time.time() - start_time

  #     return elapsed_time

  # t1 = time_operation(unique_ngrams_df, "Before Repartitioning")
  # t2 = time_operation(balanced_df, "After Repartitioning")

  return test_df

# input_df.show()

## end your edits here =================
###!@4 END ANSWER SET 4

###!@5 START ANSWER SET EVALUATION
# ========== *** DO NOT MODIFY *** ========== #

print("-----------------------------ENGLISH--------------------------")
ans0_output_df = question_0(input_df)
ans0_output_df.cache().show()
print(ans0_output_df.count())

# ans0_output_df.write.mode("overwrite").csv("ans0_output_df.csv")

print("--------------------------URL FILTER---------------------------------")
blacklist_terms = ["cryptojacking", "gambling", "stalkerware", "mixed_adult"]
ans1_1_output_df = question_1_1(ans0_output_df, blacklisted_urls_df, blacklist_terms)
ans1_1_output_df.cache().show()
print(ans1_1_output_df.count())
# ans1_1_output_df.write.mode("overwrite").csv("ans1_1_output_df.csv")

print("--------------------------WORDS FILTER---------------------------------")
ans1_2_output_df = question_1_2(ans1_1_output_df, banned_words_df)
ans1_2_output_df.cache().show()
print(ans1_2_output_df.count())
# ans1_2_output_df.write.mode("overwrite").csv("ans1_2_output_df.csv")

print("--------------------------PREPROCESSING FILTER---------------------------------")
ans2_output_df = question_2(ans1_2_output_df)
ans2_output_df.cache().show()
print(ans2_output_df.count())
# ans2_output_df.write.mode("overwrite").csv("ans2_output_df.csv")

print("--------------------------5-GRAMS FILTER---------------------------------")
ans3_1_output_df = question_3_1(ans2_output_df)
ans3_1_output_df.cache().show()
print(ans3_1_output_df.count())
# ans3_1_output_df.write.mode("overwrite").csv("ans3_1_output_df.csv")

print("--------------------------DEDUP---------------------------------")
ans3_2_output_df = question_3_2(ans3_1_output_df)
ans3_2_output_df.cache().show()
print(ans3_2_output_df.count())
# ans3_2_output_df.write.mode("overwrite").csv("ans3_2_output_df.csv")

# ans4_output = question_4(ans3_1_output_df)
# ans4_output = question_4(ans3_1_output_df)
# ans4_output

# ========== *** DO NOT MODIFY *** ========== #
###!@5 END ANSWER SET EVALUATION
