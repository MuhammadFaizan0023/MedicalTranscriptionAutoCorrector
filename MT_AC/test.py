from pipeline import AutocorrectPipeline

pipeline = AutocorrectPipeline(vocab_freq_path="./Data Files/merged_vocab_updated.pkl")
vocab_set = "./Data Files/merged_vocab_updated.pkl"
test = "then, 1% lidocaine was ued for anesthesia."
print(pipeline.correct_sentence(test,vocab_set))