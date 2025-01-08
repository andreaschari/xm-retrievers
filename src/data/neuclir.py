import sys
import csv
import pathlib
from itertools import chain
from os.path import exists, join
from typing import Optional
import ir_datasets
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    from src.utils.common import NEUCLIR_LANGUAGES
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.common import NEUCLIR_LANGUAGES

class NeuCLIRColbertLoader:
    def __init__(
        self, 
        lang: str,  # Language in which neuCLIR is loaded.
        load_test: Optional[str] = False, # Whether to load the trec test data.
        data_folder: Optional[str] = 'data/neuclir',  # Folder in which to save the downloaded datasets.
    ):
        assert lang in NEUCLIR_LANGUAGES.keys(), f"Language {lang} not supported."
        self.lang = lang
        self.data_folder = data_folder
        self.load_test = load_test
    
    def run(self):
        data_filepaths = {}
        # neuCLIR trec querie set, because of the rest of the codebase you have to manually set the year here.
        year = "2023" 
        # Load collection of passages.
        dataset = ir_datasets.load(f"neuclir/1/{self.lang}/trec-{year}")
        # Ensure the data folder exists
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        # Load collection of passages.
        save_path = join(self.data_folder, f"{self.lang}_collection.tsv")
        if not exists(save_path):
            with open(save_path, 'w', newline='') as fOut:
                writer = csv.writer(fOut, delimiter='\t')
                for sample in dataset.docs_iter():
                    writer.writerow([sample.doc_id, sample.title.replace('\n', ' ').replace('\r', ' ') + ' ' + sample.text.replace('\n', ' ').replace('\r', ' ')])
        
        # create a file with the mapping from doc_id to integer.
        save_path = join(self.data_folder, f"{self.lang}_doc_id_mapping.tsv")
        if not exists(save_path):
            with open(save_path, 'w', newline='') as fOut:
                writer = csv.writer(fOut, delimiter='\t')
                for i, sample in enumerate(dataset.docs_iter()):
                    writer.writerow([sample.doc_id, i])
        logging.info(f"Saved doc_id mapping for {self.lang}")

        # edit the collection file to use the new doc_id.
        collection = []
        with open(join(self.data_folder, f"{self.lang}_collection.tsv"), 'r') as fIn:
            reader = csv.reader(fIn, delimiter='\t')
            for row in reader:
                collection.append(row)
        
        new_collection_path = join(self.data_folder, f"{self.lang}_collection_updated.tsv")
        if not exists(new_collection_path):
            with open(new_collection_path, 'w', newline='') as fOut:
                writer = csv.writer(fOut, delimiter='\t')
                for i, row in enumerate(collection):
                    row[0] = i
                    writer.writerow([i, row[1]])
            logging.info(f"Saved updated collection file for {self.lang} to {new_collection_path}")

        data_filepaths['collection'] = new_collection_path

        # Load test queries.
        save_path = join(self.data_folder, f"{self.lang}_queries.trec-{year}.tsv")
        with open(save_path, 'w', newline='') as fOut:
            writer = csv.writer(fOut, delimiter='\t')
            for sample in dataset.queries_iter():
                writer.writerow([sample.query_id, sample.title.replace('\n', ' ').replace('\r', ' ')])
        data_filepaths['test_queries'] = save_path

        # Load test qrels.
        save_path = join(self.data_folder, f"{self.lang}_qrels.trec-{year}.tsv")
        if not exists(save_path):
            with open(save_path, 'w', newline='') as fOut:
                writer = csv.writer(fOut, delimiter='\t')
                for sample in dataset.qrels_iter():
                    writer.writerow([sample.query_id, 0, sample.doc_id, sample.relevance])
        
        # edit the qrels file to use the new doc_id
        qrels = []
        with open(join(self.data_folder, f"{self.lang}_qrels.trec-{year}.tsv"), 'r') as fIn:
            reader = csv.reader(fIn, delimiter='\t')
            for row in reader:
                qrels.append(row)
        
        new_qrels_path = join(self.data_folder, f"{self.lang}_qrels_updated.trec-{year}.tsv")
        with open(new_qrels_path, 'w', newline='') as fOut:
            with open(join(self.data_folder, f"{self.lang}_doc_id_mapping.tsv"), 'r') as fIn:
                doc_id_mapping = {}
                reader = csv.reader(fIn, delimiter='\t')
                for row in reader:
                    doc_id_mapping[row[0]] = row[1]
                writer = csv.writer(fOut, delimiter='\t')
                for row in qrels:
                    row[2] = doc_id_mapping[row[2]]
                    writer.writerow(row)
        logging.info(f"Saved updated qrels file for {self.lang} to {new_qrels_path}")
        data_filepaths['test_qrels'] = new_qrels_path
            
        logging.info(f"Loaded neuclir data for {self.lang}")
        return data_filepaths
