# 🚀 AI-Driven Hyper-Personalization & Recommendations

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
This problem statement is about Generative AI-driven solution that enhances hyper-personalization by analyzing customer profiles, social media activity, purchase history, sentiment data and demographic details and also provide actionable insights for business to optimize customer engagement.

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
By leveraging AI-driven data analysis, we aim to create a system that understands user preferences more accurately, providing tailored recommendations in real-time.This project is motivated by the potential of AI to bridge the gap between user intent and content delivery, ultimately improving customer experience and business efficiency.

## ⚙️ What It Does
•	Personalized Recommendations: Uses a RAG-based LLM system to generate targeted offers.
•	Data Ingestion & Preprocessing: Multi-Modal  Learning to enhance recommendation quality by incorporating structured and unstructured data sources.
•	Ethical AI & Fairness Checks: Integrates AI Fairness 360 to detect bias in recommendations.

## 🛠️ How We Built It
Step-by-step code implementation

Step 1: Install dependencies

Step 2: Data Preparation
	a. Generate Synthetic Data
 	b. Data Preprocessing
	c. Multi-Modal Feature Engineering

Step 3: Build a RAG-Based Recommender
	a. Generate Vector embeddings using all-mpnet-base-v2
	b. Set Up Vector Database (FAISS)
	c. Retrieve Recommendations with RAG using Mistral (LLM)	

Step 4: Streamlit Frontend
	a. Launch Interactive UI

Step 5: Ethical Checks
	a. Bias Detection

Step 6 : Benchmarking

## 🚧 Challenges We Faced and how we tackled
•	Dataset limitations as the dataset were readily available we used artificial data synthesizer to generate the data using CTGANSynthesizer and SingleTableMetadata
•	Multi-Modal Data Handling: handling structured and unstructured text data.
•	Efficient Retrieval in RAG: Used all-mpnet-base-v2 over distilbert-base-uncased for better embeddings and FAISS for faster retrieving of index.
•	Used Mistral or GPT2 for better recommendations 
•	Ethical Considerations: Ensuring fairness in recommendations.
•	LLM Constraints: Avoiding paid APIs and focusing on open-source models.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/your-repo.git
   ```
2. Install dependencies  
   ```sh
   npm install  # or pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
   npm start  # or python app.py
   ```

## 🏗️ Tech Stack
Programming Language: Python

Frameworks & Libraries:
	•	Synthetic Data Generation: SDV (CTGANSynthesizer)
	•	Feature Engineering: Pandas, NumPy, NLTK, Transformers,Huggingface, sentence transformers 
 		(distilbert-base-uncased,all-       mpnet-base-v2)
	•	RAG-Based Recommendation: FAISS, LangChain, GPT-2, Mistral
	•	Frontend: Streamlit
	•	Bias Detection: AI Fairness 360
	•	Benchmarking

## 👥 Team
- Mahesh Gain - [GitHub](#) | [LinkedIn](#)
- Dhanalakshmi Rajapandiyan - [GitHub](#) | [LinkedIn](#)
- Ravali Verghese - [GitHub](#) | [LinkedIn](#)
- Smita Singh  - [GitHub](#) | [LinkedIn](#)
- Sathyapriya Varadharaj - [GitHub](#) | [LinkedIn](#)
