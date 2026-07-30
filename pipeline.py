from symspellpy import SymSpell, Verbosity
import joblib
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

from huggingface_hub import constants

class AutocorrectPipeline:
   
    print("HF_HUB_CACHE resolved to:", constants.HF_HUB_CACHE)
    def __init__(self, vocab_freq_path="merged_vocab_updated.pkl", max_dictionary_edit_distance=2, prefix_length=7):

        print(f"Loading vocabulary from {vocab_freq_path}...")
        self.merged_freq = joblib.load(vocab_freq_path)
        self.vocab_set = set(self.merged_freq.keys())

        print("Building SymSpell dictionary...")
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_dictionary_edit_distance,
            prefix_length=prefix_length
        )
        for word, freq in self.merged_freq.items():
            self.sym_spell.create_dictionary_entry(word, freq)

        print("Loading BERT model...")
        try:
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            print("Loaded Bio_ClinicalBERT")
        except Exception as e:
            print("Bio_ClinicalBERT unavailable, falling back to bert-base-uncased:", e)
            model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.model.eval()
        self.model_name = model_name
        print(f"Pipeline ready. Using model: {model_name}")

        # Candidate generation
    def get_candidates(self, word, max_edit_distance=2, max_candidates=10, min_len=3):
        if len(word) < min_len:
            return []
        """Return candidate corrections for a single word using SymSpell."""
        suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
        filtered = [s.term for s in suggestions if len(s.term) >= min_len]
        return filtered[:max_candidates]


    def flag_and_get_candidates(self, sentence, vocab_set):
        """For each word in the sentence, flag if misspelled and get SymSpell candidates."""
        words = re.findall(r"[a-zA-Z]+", sentence)
        flagged = {}

        for i, word in enumerate(words):
            clean_word = word.lower()
            if clean_word not in vocab_set:
                flagged[i] = self.get_candidates(clean_word)

        return flagged
    def score_candidate(self, words, position, candidate):
        """
        Returns BERT's probability of `candidate` filling the masked position.
        """
        masked_words = words.copy()
        masked_words[position] = self.tokenizer.mask_token
        masked_sentence = " ".join(masked_words)

        inputs = self.tokenizer(masked_sentence, return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]

        if len(mask_token_index) == 0:
            return 0.0  # tokenization mismatch, skip

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        mask_logits = logits[0, mask_token_index[0], :]
        probs = torch.softmax(mask_logits, dim=-1)

        candidate_ids = self.tokenizer.encode(candidate, add_special_tokens=False)

        if len(candidate_ids) != 1:
            # multi-subword candidate — approximate using the first subword's probability
            return probs[candidate_ids[0]].item()

        return probs[candidate_ids[0]].item()

    def correct_word(self, words, position, candidates):
        """
        Given a sentence, the flagged word's position, and Stage 3 candidates,
        return the candidate BERT scores highest.
        """
        #words = sentence.split()
    
        if not candidates:
            return words[position]  # no candidates, leave as-is

        scores = [(cand, self.score_candidate(words, position, cand)) 
                for cand in candidates]
    
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]  # best candidate

    '''def correct_sentence(self, sentence, vocab_set):
        flagged = self.flag_and_get_candidates(sentence, vocab_set) 
        words = re.findall(r"[a-zA-Z]+", sentence) 
    
        corrected_words = words.copy()
        for position, candidates in flagged.items():
            best = self.correct_word(words, position, candidates)
            corrected_words[position] = best
    
        return " ".join(corrected_words)'''
    def correct_sentence(self, sentence):
        # Capture words AND non-word chunks (punctuation, numbers, whitespace) separately
        tokens = re.findall(r"[a-zA-Z]+|[^a-zA-Z]+", sentence)

        # Build a parallel list of just the alphabetic words, preserving their token index
        word_positions = [i for i, t in enumerate(tokens) if re.fullmatch(r"[a-zA-Z]+", t)]
        words_only = [tokens[i] for i in word_positions]

        # Run flagging on the words_only list
        flagged = {}
        for local_i, word in enumerate(words_only):
            clean_word = word.lower()
            if clean_word not in self.vocab_set:
                flagged[local_i] = self.get_candidates(clean_word)

        for local_i, candidates in flagged.items():
            best = self.correct_word(words_only, local_i, candidates)
            # Preserve original capitalization pattern
            if words_only[local_i][0].isupper():
                best = best.capitalize()
            token_index = word_positions[local_i]
            tokens[token_index] = best

        return "".join(tokens)