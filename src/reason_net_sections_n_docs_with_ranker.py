import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import huggingface_hub
import itertools
import json
import numpy as np
import re
import pandas as pd
import gc
from logger import logger

cache_location = '.cache'
os.environ['HF_HOME'] = cache_location
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_location
os.environ['HF_HOME'] = cache_location
os.environ['TRANSFORMERS_CACHE'] = cache_location
os.environ['HF_DATASETS_CACHE'] = cache_location
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_location
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open("model_triples.json", "r") as f:
    llm_data = json.load(f)

for part_name, triple in llm_data.items():
    logger.info(f"Processing {part_name} with models: {triple}")
    PART_NAME = part_name
    TRIPLE = triple

class ReasonNetWithSectionsNDocs:
    """
    A small 'ReasonNet' that:
      - Classifies the domain from the user prompt
      - Generates a short 'reasoning snippet'
      - Produces a structured plan
    """

    def __init__(self, model, tokenizer, device="cpu"):
        #logger.info(f"Loading ReasonNet model: {model_name}")
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = device
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True
        )

    def generate_reasoning_snippet(self, user_prompt: str, temp_value: float, reasoning_approach: str) -> str:
        """
        Generate a short 'reasoning snippet' with the smaller model.
        We'll keep it short to simulate an intermediate chain-of-thought.
        """
        if reasoning_approach == 'doc':
            reasoning_prompt = f"""
                Act as an expert Python programmer and logical driven code reasoner. Your task is to generate a code reasoning snippet in a detailed breakdown.

                User Request: {user_prompt}

                Please provide your response in the following structured in a markdown format:

                The documentation should include the following sections:
                1. Problem Description: How to approach the problem and what the main challenges are.
                2. Domain-Specific Libraries : The field(s) or context (e.g., math, computer vision, NLP, remote sensing)
                3. Algorithm Design / Pseudocode : Provide the pseudocode  implementation, of your reasoning, it need not be a full code
                4. Implementation Considerations : Edge cases and other things that needs to be considered in the implementation
                5. References (optional)

                Ensure your reasoning is detailed and explains the 'why' behind your pseudocode decisions. Always ensure the reasoning snippet comes before pseudocode, and that the reasoning matches the pseudocode.
                Output only the reasoning snippet.
                """
        elif reasoning_approach == 'sections':
            reasoning_prompt = f"Can you give a short and precise pseudocode on how to solve the following:\n\n Request: {user_prompt}. Output only the reasoning snippet."
        # Generate a short snippet
        out = self.pipeline(
            reasoning_prompt,
            num_return_sequences=1,
            temperature=temp_value,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=256
        )[0]["generated_text"]

        return out

    def classify_domain(self, user_prompt: str) -> str:
        """
        Naive domain classification using keywords.
        In practice, you could do a more sophisticated approach.
        """        
        prompt_lower = user_prompt.lower()
        classify_domain_prompt = f"Classify the scientific domain of the user's request, examples are : Medicine, Econimics, Politics, Software Engineering of the following request:\n\n Request: {prompt_lower}\n Output only the scientific domain."

        domain_out = self.pipeline(
            classify_domain_prompt,
            num_return_sequences=1,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=256
        )[0]["generated_text"]

        domain_out = domain_out.split("Scientific-Domain:")[-1].strip()
        domain_out = domain_out.split("\n")[0].strip()  # Get the first line after "Domain:"
        return domain_out
    
    def suggest_libraries(self, domain: str, user_prompt: str) -> list:
        """
        Suggest libraries based on the domain.
        """
        suggested_libraies_prompt = f"Suggest Python libraries in a list for suggested scientific domain: \n {domain} to solve the task: \n{user_prompt}  \n Libraries-list:"
        
        suggested_libraries = self.pipeline(
            suggested_libraies_prompt,
            num_return_sequences=1,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=1024
        )[0]["generated_text"]

        libraries = suggested_libraries.split("Libraries-list:")[-1].strip()
        libraries = libraries.split(",")
        return libraries

    def produce_structured_plan(self, user_prompt: str, domain: str, reasoning_text: list, libraries: list) -> dict:
        """
        Create a simple plan with domain, user_prompt, reasoning, suggested approach.
        """
        plan = {
            "domain": domain,
            "user_prompt": user_prompt,
            "reasoning_snippet": reasoning_text,
            "libraries": libraries
        }
        return plan
    
    def run_reasoning_pipeline(self, user_prompt: str, reasoning_approach: str) -> dict:
        """
        1) Classify domain
        2) Generate short reasoning snippet
        3) Produce structured plan
        """
        if reasoning_approach == 'doc':
            plan = self.generate_reasoning_snippet(user_prompt, 0.1,reasoning_approach)
        elif reasoning_approach == 'sections':
            domain = self.classify_domain(user_prompt)
            snippet = self.generate_reasoning_snippet(user_prompt, 0.1, reasoning_approach)
            libraries = self.suggest_libraries(domain, user_prompt)
            plan = self.produce_structured_plan(user_prompt, domain, snippet, libraries)
        return plan

class CodeGenerator:
    """
    Uses a LLM chat model to generate code. 
    This is a minimal example. In practice, you'd follow the chat prompt format 
    more carefully for Llama 2.
    """

    def __init__(self, model, tokenizer, device="cuda:0"):
        """
        Initialize the Llama Code Generator with a local model directory.
        :param model_path: Path to the local directory containing the model files.
        :param device: Device to load the model on (e.g., 'cuda:0' or 'cpu').
        """
        device = device if torch.cuda.is_available() else "cpu"
        self.model = model
        self.tokenizer = tokenizer
        # Build a pipeline for text-generation or chat

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True
        )

    def generate_code(self, plan: dict) -> str:
        """
        Create a prompt for the LLM model for generating code.
        """
        
        user_prompt = f"""
            You are an expert scientific programmer.

            Based on the following technical documentation, generate Python code that accomplishes the task as described.

            Please use any domain-specific libraries mentioned, follow the pseudocode carefully, and incorporate any noted implementation considerations.

            Documentation:
            {plan}

            Please output only the final Python code block — no explanation or extra text.
            """

        model_max_length = getattr(self.tokenizer, 'model_max_length', 2048)
        if model_max_length > 100000:  # Some tokenizers return very large defaults
            model_max_length = 2048  # fallback to reasonable default
        
        max_new_tokens = 1024


        output = {}

        try:
            result = self.pipeline(
                user_prompt,
                num_return_sequences=1,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                max_new_tokens=max_new_tokens
            )
            if not result or not isinstance(result, list) or len(result) == 0 or "generated_text" not in result[0]:
                logger.info("Code generation failed: No output from model.")
                return ""
            print(result)
            full_output = result[0]["generated_text"]
            # Remove the prompt from the output
            if user_prompt in full_output:
                output = full_output.replace(user_prompt, '').strip()
            else:
                # fallback in case prompt is not exactly matched
                output = full_output.split("Documentation:")[-1].strip()
        except Exception as e:
            logger.info(f"Code generation failed: {e}")
            output = ""
        return output

def generate_model_pairs():
    # model_id = "meta-llama/Meta-Llama-3-8B-instruct"
    # models_ls = ["EleutherAI/gpt-neo-2.7B", "Qwen/CodeQwen1.5-7B-Chat", "EleutherAI/gpt-neo-1.3B","microsoft/wavecoder-ultra-6.7b"]
    # models_ls = ["microsoft/wavecoder-ultra-6.7b","meta-llama/Meta-Llama-3-8B-instruct", "EleutherAI/gpt-neo-1.3B"]
    # pairs = list(itertools.product(models_ls, repeat=2))
    # pairs = [pair for pair in pairs if pair[0] == pair[1]]
    # pairs = [["meta-llama/Meta-Llama-3-8B-instruct","meta-llama/Meta-Llama-3-8B-instruct"]]


    triples = [["microsoft/wavecoder-ultra-6.7b", "microsoft/wavecoder-ultra-6.7b", "meta-llama/Meta-Llama-3-8B-instruct"]]

    return triples

def authenticate(token, auth_switch):
    """
    Login to Hugging Face Hub using the provided token.
    :param token: Hugging Face token for authentication.
    """
    if auth_switch == 'login':
        try:
            huggingface_hub.login(token)
            logger.info("Logged in to Hugging Face Hub successfully.")
        except Exception as e:
            logger.info(f"Error logging in: {e}")
            raise e
    elif auth_switch == 'logout':
        try:
            huggingface_hub.logout()
            logger.info("Logged out from Hugging Face Hub successfully.")
        except Exception as e:
            logger.info(f"Error logging out: {e}")
            raise e
    else:
        logger.info("Invalid auth_switch value. Use 'login' or 'logout'.")
        raise ValueError("Invalid auth_switch value. Use 'login' or 'logout'.")

class CodeRanker:
    def __init__(self, model, tokenizer, device="cpu"):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = device
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True
        )

    def compare_codes(self, user_prompt, code_1, code_2):
        # rank_prompt = f'''As an expert code reviewer and in the area of geospatial work.Look at the question {user_prompt} and Compare code_1 {code_1} and \n code_2 {code_2}. 
        # A better code that is readable and also correct. Output 1 if code_1 is better and 0 if code_2 is better. Output only the number 1 or 0.'''
        rank_prompt = f''' As an expert code reviewer in geospatial programming, compare the following:
            Question: {user_prompt}

            Code_1:
            {code_1}

            Code_2:
            {code_2}

            A better code is one that is correct and readable.

            Your task:
            Output only one value with no explanation.
            - 1 if code_1 is better.
            - 0 if code_2 is better.
            - 0.5 if they are almost the same.

            You must output exactly one token: 1 or 0 or 0.5.'''

        response = self.pipeline(
            rank_prompt,
            num_return_sequences=1,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=256
        )[0]["generated_text"]

        # Extract strictly 1, 0, or 0.5
        print(f"Single response ----> {response}")
        match = re.search(r"\b(1|0|0\.5)\b", response)
        outcome =  match.group(1) if match else None
        print(f"Outcome ----> {outcome}")
        return float(outcome)

    def get_code_outcomes(self, user_prompt, generated_responses):
        index_of_responses = np.arange(len(generated_responses))
        pairs = list(itertools.combinations(index_of_responses, 2))
        pairs_output = []
        for pair in pairs:
            code_1 = generated_responses[pair[0]]
            code_2 = generated_responses[pair[1]]
            print(f"Comparing code_1: {code_1} with code_2: {code_2}")
            rank_result = self.compare_codes(user_prompt, code_1, code_2)
            logger.info(f"Rank result for pair {pair}: {rank_result}")
            pairs_output.append(rank_result)

        pairs_with_rank_df = pd.DataFrame(pairs, columns=["Anchor", "Anchor_pair"])
        pairs_with_rank_df['outcome'] = pairs_output

        pairs_with_rank_df.to_excel("pairs_with_rank.xlsx", index=False)

        return pairs_with_rank_df

    def rank_codes(self, pairs_with_rank_df, generated_responses):
        ranked_data = pairs_with_rank_df.copy()
        # Get all unique indices from both anchor and anchor_pair columns
        unique_indices = sorted(set(ranked_data['Anchor'].unique()) | set(ranked_data['Anchor_pair'].unique()))
        n_indices = len(unique_indices)
        
        # Create mapping from indices to array positions
        index_to_pos = {idx: pos for pos, idx in enumerate(unique_indices)}
        
        # Initialize score array
        scores = np.zeros(n_indices)
        
        # Calculate scores from pairwise results
        for _, row in ranked_data.iterrows():
            anchor_pos = index_to_pos[row['Anchor']]
            anchor_pair_pos = index_to_pos[row['Anchor_pair']]
            outcome = row['outcome']
            
            # Add score for anchor based on outcome
            scores[anchor_pos] += outcome
            # Add complementary score for anchor_pair
            scores[anchor_pair_pos] += (1.0 - outcome)
        
        # Handle ties in ranking
        # Get unique scores and sort them in descending order
        unique_scores = np.unique(scores)[::-1]
        
        # Create rankings array, handling ties
        rankings = np.zeros(n_indices, dtype=int)
        current_rank = 1
        
        for score in unique_scores:
            # Find all indices with this score
            indices_with_score = np.where(scores == score)[0]
            
            # Assign the same rank to all indices with the same score
            for idx in indices_with_score:
                rankings[idx] = current_rank
            
            # Update rank for next group (skip tied positions)
            current_rank += len(indices_with_score)
        
        results_df = pd.DataFrame({
            'code': unique_indices,
            'score': scores,
            'rank': rankings
        }).sort_values('rank')

        results_df['code'] = results_df['code'].apply(lambda x: generated_responses[x])
        results_df.sort_values('rank', inplace=True)
        results_df.to_excel("ranked_codes.xlsx", index=False)
        return results_df
        
    def get_correct_code(self, ranked_data):
        if ranked_data.empty:
            return None
        top_code = ranked_data.iloc[0]['code']
        return top_code
    
    def main(self,user_prompt, generated_codes):
        pairs_with_rank_df = self.get_code_outcomes(user_prompt, generated_codes)
        ranked_data = self.rank_codes(pairs_with_rank_df, generated_codes)
        correct_code = self.get_correct_code(ranked_data)
        return correct_code


def load_unique_models(model_pairs):
    """Load each unique model only once and return a dictionary of loaded models."""
    unique_models = set()
    for reasoner, coder, ranker in model_pairs:
        unique_models.update([reasoner, coder, ranker])
    
    loaded_models = {}
    
    for model_name in unique_models:
        logger.info(f"Loading model: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            loaded_models[model_name] = {
                'model': model,
                'tokenizer': tokenizer
            }
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e
    
    return loaded_models

if __name__ == "__main__":
    with open('../hf_key.txt', 'r') as file:
        key = file.readline()
    authenticate(key, 'login')
    
    model_pairs = TRIPLE
    print(model_pairs)
    # Sample tasks
    tasks = [
         "Given the red and near-infrared (NIR) reflectance values of a pixel as floating-point numbers, write Python code to compute the NDVI: (NIR - Red) / (NIR + Red). Return 0 if the denominator is zero, and classify the pixel as ‘vegetation’ if NDVI > 0.3, ‘urban’ if NDVI < 0.2, and ‘transition’ otherwise.",
         "Given the spectral band values for a pixel: [Red = 0.15, Green = 0.35, Blue = 0.20, NIR = 0.60], write Python code to compute NDVI and classify the pixel as ‘vegetation’, ‘water’, or ‘urban’ based on NDVI thresholds.",
         "Write a Python function that takes the spectral band values of a pixel (at minimum Red and NIR) as input, computes NDVI, and classifies the land cover as ‘vegetation’, ‘water’, or ‘urban’ using defined thresholds.",
         "Given a Sentinel-2 GeoTIFF file, write  a Python code to compute NDVI for every pixel and classify each pixel as ‘urban’ if NDVI < 0.2, ‘vegetation’ if NDVI > 0.3, and ‘transition’ otherwise.",
         "Write Python code to compute NDVI from a GeoTIFF file with Red and NIR bands, and classify each pixel into ‘vegetation’, ‘urban’, or ‘transition’ using thresholds of 0.3 and 0.2. Save the output as a new classified raster file.",
         "Given an Excel sheet with columns for Red and NIR reflectance values, write Python code to compute NDVI for each row and assign a land cover category (‘vegetation’, ‘urban’, or ‘transition’) based on NDVI thresholds.",
         "Given a Sentinel-2 GeoTIFF image containing Red and NIR bands, write Python code to compute NDVI and classify the land cover into ‘vegetation’, ‘urban’, or ‘water’."
    ]

    data_dict = {}
    data_dict['reasoner'] = []
    data_dict['coder'] = []
    data_dict['plan'] = []
    data_dict['code'] = []
    data_dict['question'] = []
    data_dict['reasoning_type'] = []
    data_dict['reasoning_approach'] = []
    data_dict['ranker'] = []

    with open("manually_carafted_reasoning.json", "r") as f:
        manually_crafted_reasoning = json.load(f)

    with open("no_reasoning.json", "r") as f:
        no_reasoning = json.load(f)

    for reasoner, coder, ranker in model_pairs:
        logger.info(f"Processing pair: Reasoner - {reasoner}, Coder - {coder}, Ranker - {ranker}")

        # Clear any existing models from memory first
        torch.cuda.empty_cache()
        gc.collect()

        if reasoner == coder:
            logger.info(f"Processing self-pair: {reasoner}")
            tokenizer = AutoTokenizer.from_pretrained(reasoner)
            model = AutoModelForCausalLM.from_pretrained(
                reasoner,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",  # Let transformers decide the best placement
                low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
            )
            reasoner_model = model
            code_model = model
            
            reason_net = ReasonNetWithSectionsNDocs(model=reasoner_model, tokenizer=tokenizer, device="auto") 
        else:
            # Load reasoner model first
            logger.info(f"Loading reasoner model: {reasoner}")
            reasoner_tokenizer = AutoTokenizer.from_pretrained(reasoner)
            reasoner_model = AutoModelForCausalLM.from_pretrained(
                reasoner, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            reason_net = ReasonNetWithSectionsNDocs(model=reasoner_model, tokenizer=reasoner_tokenizer, device="auto")
            
        # Process reasoning tasks first
        reasoning_plans = []
        for task in tasks:
            logger.info(f"Generating reasoning for: {task}")
            try:
                plan = reason_net.run_reasoning_pipeline(task,'doc')
                reasoning_plans.append((task, plan, 'automated', 'doc'))
            except Exception as e:
                raise ValueError(f"Reasoner Error with doc: {e}")
            
            try:
                plan = reason_net.run_reasoning_pipeline(task,'sections')
                reasoning_plans.append((task, plan, 'automated', 'sections'))
            except Exception as e:
                raise ValueError(f"Reasoner Error with sections: {e}")

        # Add manually crafted reasoning
        for plan in manually_crafted_reasoning:
            user_prompt = plan['user_prompt']
            reasoning_plans.append((user_prompt, plan, 'manual', None))

        for plan in no_reasoning:
            user_prompt = plan['user_prompt']
            reasoning_plans.append((user_prompt, plan, 'no_reasoning', None))

        
        if reasoner == coder:
            code_tokenizer = tokenizer
            code_gen = CodeGenerator(model=code_model, tokenizer=code_tokenizer, device="auto")
        else:
            # Now load coder model
            logger.info(f"Loading coder model: {coder}")
            code_tokenizer = AutoTokenizer.from_pretrained(coder)
            code_model = AutoModelForCausalLM.from_pretrained(
                coder, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            code_gen = CodeGenerator(model=code_model, tokenizer=code_tokenizer, device="auto")

        if ranker == coder:
            ranker_model = CodeRanker(model=code_model, tokenizer=code_tokenizer, device="auto")
        elif ranker == reasoner:
            ranker_model = CodeRanker(model=reasoner_model, tokenizer=reasoner_tokenizer, device="auto")

            # Clean up reasoner model if not needed anymore
            del reason_net
            del reasoner_model
            torch.cuda.empty_cache()
            gc.collect()
        
        else:
            logger.info(f"Loading ranker model: {ranker}")
            ranker_tokenizer = AutoTokenizer.from_pretrained(ranker)
            ranker_model_instance = AutoModelForCausalLM.from_pretrained(
                ranker, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
            ranker_model = CodeRanker(model=ranker_model_instance, tokenizer=ranker_tokenizer, device="auto")

        # Process all code generation tasks
        iterations = 5
        for task, plan, reasoning_type, reasoning_approach in reasoning_plans:
            generated_codes = []
            if plan:  # Only process if we have a valid plan
                for r in range(iterations):
                    code_solution = code_gen.generate_code(plan)
                    generated_codes.append(code_solution)

            best_ranked_code = ranker_model.main(task, generated_codes)

            data_dict['reasoner'].append(reasoner)
            data_dict['coder'].append(coder)
            data_dict['ranker'].append(ranker)
            data_dict['plan'].append(plan)
            data_dict['code'].append(best_ranked_code)
            data_dict['question'].append(task)
            data_dict['reasoning_type'].append(reasoning_type)
            data_dict['reasoning_approach'].append(reasoning_approach)

            
        # Clean up models after processing each pair
        del reason_net
        del code_gen
        del reasoner_model
        del code_model
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save intermediate results
        with open(f'data_output_reasoning_with_none_reason_manual_{PART_NAME}.json', 'w') as f:
            json.dump(data_dict, f, indent=4)