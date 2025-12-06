"""
Generate massive synthetic dataset for RAG evaluation
Uses LLM to create diverse question variations
"""

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

from langsmith import Client
from langchain_openai import ChatOpenAI
import json

client = Client()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# Your 11 textbook topics
TOPICS = {
    "economics": [
        "microeconomics", "macroeconomics", "supply and demand", 
        "inflation", "GDP", "fiscal policy", "monetary policy",
        "market structures", "opportunity cost", "trade"
    ],
    "anatomy": [
        "muscular system", "skeletal system", "nervous system",
        "cardiovascular system", "respiratory system", "digestive system",
        "human organs", "cells", "tissues", "body regions"
    ],
    "political_science": [
        "democracy", "authoritarianism", "political parties",
        "elections", "constitution", "federalism", "civil rights",
        "international relations", "political ideologies"
    ],
    "business": [
        "entrepreneurship", "marketing", "management",
        "finance", "accounting", "operations", "human resources",
        "business ethics", "strategic planning"
    ],
    "sociology": [
        "social stratification", "culture", "socialization",
        "deviance", "social institutions", "race and ethnicity",
        "gender", "social movements", "globalization"
    ],
    "machine_learning": [
        "supervised learning", "unsupervised learning", "neural networks",
        "deep learning", "gradient descent", "overfitting",
        "regularization", "classification", "regression", "clustering"
    ],
    "data_science": [
        "statistics", "probability", "data visualization",
        "hypothesis testing", "regression analysis", "sampling",
        "correlation", "data cleaning", "feature engineering"
    ],
    "linear_algebra": [
        "vectors", "matrices", "linear transformations",
        "eigenvalues", "eigenvectors", "vector spaces",
        "determinants", "matrix operations", "orthogonality"
    ]
}

# Question templates
TEMPLATES = [
    "What is {concept}?",
    "Explain {concept} in simple terms.",
    "How does {concept} work?",
    "What are the key principles of {concept}?",
    "Give an example of {concept}.",
    "Why is {concept} important?",
    "Compare and contrast {concept} with related concepts.",
    "What are the applications of {concept}?",
    "Describe the main components of {concept}.",
    "What problems does {concept} solve?",
]

def generate_questions_for_topic(topic: str, concepts: list) -> list:
    """Generate diverse questions for a topic"""
    questions = []
    
    for concept in concepts:
        for template in TEMPLATES:
            questions.append({
                "question": template.format(concept=concept),
                "topic": topic,
                "concept": concept
            })
    
    return questions

def generate_llm_variations(base_questions: list, num_variations: int = 3) -> list:
    """Use LLM to create natural variations of questions"""
    
    variations = []
    
    for q in base_questions[:50]:  # Limit for API cost
        prompt = f"""Generate {num_variations} natural variations of this question. 
Make them sound like real student questions - casual, sometimes incomplete, sometimes with typos.

Original: {q['question']}

Return as JSON array: ["variation1", "variation2", "variation3"]
"""
        try:
            result = llm.invoke(prompt)
            parsed = json.loads(result.content)
            for var in parsed:
                variations.append({
                    "question": var,
                    "topic": q["topic"],
                    "concept": q["concept"],
                    "original": q["question"]
                })
        except:
            pass
    
    return variations

def create_massive_dataset():
    """Create the full synthetic dataset"""
    
    dataset_name = "synthetic_massive"
    
    # Delete if exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing.id)
        print(f"Deleted existing: {dataset_name}")
    except:
        pass
    
    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Massive synthetic dataset covering all topics"
    )
    
    all_questions = []
    
    # Generate base questions
    print("📝 Generating base questions...")
    for topic, concepts in TOPICS.items():
        questions = generate_questions_for_topic(topic, concepts)
        all_questions.extend(questions)
        print(f"   {topic}: {len(questions)} questions")
    
    print(f"\n📊 Total base questions: {len(all_questions)}")
    
    # Generate LLM variations (optional - costs money)
    generate_variations = input("\nGenerate LLM variations? (y/n): ").lower() == 'y'
    
    if generate_variations:
        print("\n🤖 Generating LLM variations...")
        variations = generate_llm_variations(all_questions)
        all_questions.extend(variations)
        print(f"   Added {len(variations)} variations")
    
    # Add negative examples (should say "I don't know")
    negative_examples = [
        {"question": "What is quantum entanglement?", "topic": "not_in_db", "concept": "physics"},
        {"question": "Who won the 2024 election?", "topic": "not_in_db", "concept": "current_events"},
        {"question": "What is the recipe for chocolate cake?", "topic": "not_in_db", "concept": "cooking"},
        {"question": "How do I fix my car engine?", "topic": "not_in_db", "concept": "automotive"},
        {"question": "What movies are playing tonight?", "topic": "not_in_db", "concept": "entertainment"},
    ]
    all_questions.extend(negative_examples)
    
    # Upload to LangSmith
    print(f"\n☁️ Uploading {len(all_questions)} examples to LangSmith...")
    
    for i, q in enumerate(all_questions):
        client.create_example(
            inputs={"question": q["question"]},
            outputs={
                "topic": q["topic"],
                "concept": q.get("concept", ""),
                "is_negative": q["topic"] == "not_in_db"
            },
            dataset_id=dataset.id
        )
        
        if (i + 1) % 100 == 0:
            print(f"   Uploaded {i + 1}/{len(all_questions)}")
    
    print(f"\n✅ Created '{dataset_name}' with {len(all_questions)} examples")
    
    # Summary
    print("\n📊 Dataset Summary:")
    for topic in set(q["topic"] for q in all_questions):
        count = sum(1 for q in all_questions if q["topic"] == topic)
        print(f"   {topic}: {count}")
    
    return dataset

if __name__ == "__main__":
    print("="*60)
    print("🔧 Massive Synthetic Dataset Generator")
    print("="*60)
    create_massive_dataset()