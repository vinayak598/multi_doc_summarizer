# summarizers.py
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline

# Extractive summarizer using LexRank
def textrank_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Abstractive summarizer using DistilBART (lightweight)
abstractive_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def abstractive_summary(text):
    if len(text.strip()) == 0:
        return "No input text provided."
    # Limit tokens to fit low RAM
    inputs = text[:1024]  
    summary = abstractive_pipeline(inputs, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']
