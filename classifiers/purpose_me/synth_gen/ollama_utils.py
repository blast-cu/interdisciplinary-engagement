from tqdm import tqdm
import time
import os
import argparse
import logging

import concurrent
import ollama
from pydantic import BaseModel

from data_utils.utils import save_file, load_file

"""
Utility functions for annotating data with an LLM.
Includes functions for loading models, generating text, and formatting data.
"""


class OllamaClient:
    """
    Class to handle the Ollama client.
    """
    class Answer(BaseModel):
        """
        Class to format the response from the LLM.
        """
        citation: str
        context: str
    
    class Messages():
        """
        Class to handle the messages for the LLM.
        """
        def __init__(self, system_prompt: str, user_head_prompt: str):
            self.system_prompt = system_prompt
            self.head_user_prompt = user_head_prompt
        
        def add_doc_prompt(self, doc_prompt: str):
            to_process = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.head_user_prompt + "\n\n" + doc_prompt}
            ]
            return to_process
    

    def __init__(self, host, port, model, seed, temperature, system_prompt, user_head_prompt, logger):
        self.logger = logger
        server_host = f"{host}:{port}"
        self.client = ollama.Client(server_host)

        self.model = model
        self.seed = seed
        self.temperature = temperature

        ollama.pull(model)

        self.options: ollama.Options = {
            "seed": seed,
            "temperature": temperature
        }

        # set the messages for client.
        self.messages = self.Messages(system_prompt, user_head_prompt)

    def chat(self, doc_prompt: str):
        """
        Chat with the LLM and check the response.
        """
        to_process = self.messages.add_doc_prompt(doc_prompt)
        
        response = self.client.chat(
            self.model,
            messages=to_process,
            options=self.options,
            format=self.Answer.model_json_schema()
        )
        
        try:
            response = self.Answer.model_validate_json(
                response.message.content
            )
            if response is not None:  # make the output serializable.
                response = response.model_dump()
            return response

        except Exception as e:
            self.logger.exception("Exception: " + str(e))
            self.logger.exception("Invalid response. Please try again.")
            return None


class Annotate:
    """
    Class for annotating text with LLM output.
    """

    def __init__(
            self, args: argparse.Namespace, 
            script_path: str, data_path: str, logger: logging.Logger,
    ):
        # set up logging.
        self.logger = logger

        logger.info("Setting Annotator variables and initializing Ollama client.")
        self.config = args

        # set the output filename if not provided.
        if args.out_filename is None:
            data_file = args.dataset.split(".")[0]
            self.config.out_filename = f"{data_file}_processed.json"

        # set directories for reading/writing data.
        self.data_path = data_path
        self.results_path = os.path.join(data_path, "results")
        os.makedirs(self.results_path, exist_ok=True)

        # load the data.
        self.prompt_data, self.docs = \
            self.load_data(script_path, data_path)

        # check for existing results.
        self.already_processed, self.docs = \
            self.handle_processed()
        
        # set the head messages for the conversations.
        system_prompt = self.prompt_data["system_prompt"]
        user_head_prompt = self.get_user_head_prompt()
        
        # initialize the ollama client.
        self.ollama_client = OllamaClient(
            args.host,
            args.port,
            args.model,
            args.seed,
            args.temperature,
            system_prompt,
            user_head_prompt,
            logger
        )

    def load_data(self, script_path: str, data_path: str) -> tuple[dict, dict]:
        """
        Load the prompt data and dataset from json.
        """
        prompt_path = os.path.join(script_path, "prompts", self.config.prompt_file)
        prompt_data = load_file(
            prompt_path, logger=self.logger
        )

        dataset_path = os.path.join(data_path, self.config.dataset)
        dataset = load_file(
            dataset_path, logger=self.logger
        )

        return prompt_data, dataset

    def handle_processed(self) -> tuple[dict, dict]:
        """
        Load docs that have already been processed and 
        remove them from the list of docs to process.
        """
        # check for existing results.
        out_file_path = os.path.join(
            self.results_path, self.config.out_filename
        )
        already_processed = {}
        to_process = self.docs
        if os.path.exists(out_file_path):
            existing_data = load_file(out_file_path, logger=self.logger)
            already_processed = existing_data["data"] if "data" in existing_data else {}
            self.logger.info(f"Found {len(already_processed)} existing results.")

            # remove already processed docs from the list.
            to_process = {doc_id: doc for doc_id, doc in self.docs.items()
                          if doc_id not in already_processed.keys()}

        return already_processed, to_process

    def get_user_head_prompt(self) -> str:
        """
        Set the head messages for the conversations.
        """
        # implement n shot prompting.
        shots = []
        head_user_prompt = ""
        if "demos" in self.prompt_data.keys() and len(self.prompt_data["demos"]) > 0:
            for demo_item in self.prompt_data["demos"]:
                user_turn = "user: " + self.prompt_data["question"] + "\n" + demo_item["text"]
                assissant_turn = "assistant: " + str(demo_item["answer"])
                shots.append(user_turn + "\n" + assissant_turn)

            head_user_prompt = "\n\n".join(shots)

        return head_user_prompt

    def format_doc(self, doc_data: dict) -> dict:
        """
        Format the doc data into an entry for processing.
        returns
            - dict: {"text": str}
        """
        # raise NotImplementedError("must write function to format doc")
        entry = {}
        entry["text"] = doc_data["text"]

        return entry

    def annotate(self, doc_prompt: str) -> dict:
        """
        Annotate the messages with the LLM. Try multiple times if the response is invalid.
        """
        max_retries = self.config.max_retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                annotation = self.ollama_client.chat(
                    doc_prompt
                )
                if annotation is not None:
                    return annotation
                else:  # retry if the response did not match the schema.
                    retry_count += 1

            except Exception as e:
                self.logger.exception("Exception: " + str(e))
                self.logger.exception("Ollama Error. Please try again.")
                retry_count += 1
        return None
    
    def process_doc(self, doc: dict) -> dict:
        """
        Process the doc with the LLM.
        """
        doc_prompt = self.prompt_data["question"] + '\n' + doc["text"]
        annotation = self.annotate(doc_prompt)
        doc["sample"] = annotation
        return doc
    
    def save_results(self, processed_data: dict):
        """
        Save the results and config details to a json file.
        """
        final_output = {}
        final_output["config"] = vars(self.config)
        final_output["time_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        final_output["prompt_data"] = self.prompt_data
        final_output["data"] = processed_data

        save_file(
            final_output, self.results_path,
            self.config.out_filename, logger=self.logger
        )

    def process_docs(self):
        """
        Process the docs in parallel.
        """

        num_workers = self.config.workers
        save_interval = self.config.save_interval

        self.logger.info("Formatting docs")

        data = {}  # dict to store formatted docs.
        for doc_id, doc_data in tqdm(self.docs.items()):
            entry = self.format_doc(doc_data)
            data[doc_id] = entry

        annotated_docs = self.already_processed
        total_docs = len(data)

        # Use a ThreadPoolExecutor for parallel processing
        self.logger.info(f"Processing docs with {num_workers} workers.")  
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_doc, doc): doc_id for doc_id, doc in data.items()}

            # Initialize tqdm progress bar to track doc processing
            with tqdm(total=total_docs) as pbar:
                processed_count = 0
                # Process docs and save at regular intervals
                for future in concurrent.futures.as_completed(futures):
                    doc_idx = futures[future]
                    try:
                        # Get the results of process_doc() for each doc
                        doc = future.result()
                        annotated_docs[doc_idx] = doc
                        processed_count += 1

                        # Update the progress bar
                        pbar.update(1)

                        # Save annotated_docs at regular intervals
                        if processed_count % save_interval == 0:
                            self.save_results(annotated_docs)
                            self.logger.info(f"Progress saved after processing {processed_count} docs.")

                    except Exception as e:
                        self.logger.exception(f"Error processing doc {doc_idx}: {e}")

        # Save the final results.
        self.save_results(annotated_docs)
