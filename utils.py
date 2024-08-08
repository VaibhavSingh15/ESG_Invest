from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")
business_areas = {
    "Product Development": ["innovation", "research", "development", "product improvement", "AI-driven design"],
    "Customer Service": ["customer support", "chatbots", "personalization", "service automation"],
    "Operations Efficiency": ["automation", "process optimization", "efficiency", "cost reduction"],
    "Marketing and Sales": ["marketing", "advertising", "sales", "customer targeting", "AI-driven campaigns"],
    "Risk Management": ["risk assessment", "fraud detection", "security", "compliance", "predictive analytics"]
}