import os
import asyncio

import streamlit as st
import requests
import json
import time
import threading
import datetime
import assemblyai as aai
from dotenv import load_dotenv
import google.generativeai as genai
from google.genai import types
import pandas as pd
import faiss
import numpy as np
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel
import logging
from typing import Dict, List, Optional, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API keys
ALPHA_VANTAGE_API_KEY = "DG91C0R2EM7HK8ZY"
ASSEMBLYAI_API_KEY = "bec2473de05746d8bcff0bd3c395f546"
GOOGLE_API_KEY = "AIzaSyAmHi2CiA4Z-K5-phmyKpd0vzgljrLNUkY"


# Initialize APIs
aai.settings.api_key = ASSEMBLYAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

# Vector store for RAG
class VectorStore:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        
    def add_texts(self, texts, embeddings):
        if len(texts) > 0:
            self.texts.extend(texts)
            self.index.add(np.array(embeddings).astype('float32'))
    
    def similarity_search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        return [self.texts[i] for i in indices[0]]

# Initialize vector store
vector_store = VectorStore()

# Agent base class
class Agent:
    def __init__(self, name):
        self.name = name
        
    async def process(self, input_data):
        raise NotImplementedError("Each agent must implement its own process method")

# API Agent for financial data
class APIAgent(Agent):
    def __init__(self):
        super().__init__("API Agent")
        
    async def process(self, input_data):
        logger.info(f"{self.name} processing: {input_data}")
        
        if "risk exposure" in input_data.lower() and "asia tech" in input_data.lower():
            # Get portfolio data
            portfolio_data = await self.get_portfolio_data()
            
            # Get earnings surprises
            earnings_data = await self.get_earnings_data()
            
            return {
                "portfolio": portfolio_data,
                "earnings": earnings_data
            }
        return {"error": "Could not process the request"}
    
    async def get_portfolio_data(self):
        # In a real implementation, this would fetch actual portfolio data
        # For this demo, we'll return mock data
        return {
            "asia_tech_allocation": 22,
            "previous_allocation": 18,
            "aum": 10000000,
            "top_holdings": [
                {"symbol": "TSM", "name": "TSMC", "allocation": 5.2},
                {"symbol": "SSNLF", "name": "Samsung", "allocation": 4.8},
                {"symbol": "9988.HK", "name": "Alibaba", "allocation": 3.5}
            ]
        }
    
    async def get_earnings_data(self):
        try:
            # Get earnings data from Alpha Vantage
            symbols = ["TSM", "SSNLF"]
            earnings_results = {}
            
            for symbol in symbols:
                url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url)
                data = response.json()
                
                if "quarterlyEarnings" in data:
                    latest_earnings = data["quarterlyEarnings"][0]
                    reported_eps = float(latest_earnings.get("reportedEPS", 0))
                    estimated_eps = float(latest_earnings.get("estimatedEPS", 0))
                    surprise_percentage = float(latest_earnings.get("surprisePercentage", 0))
                    
                    earnings_results[symbol] = {
                        "reported_eps": reported_eps,
                        "estimated_eps": estimated_eps,
                        "surprise_percentage": surprise_percentage
                    }
            
            # For Samsung, we might need to mock data as it might not be available in Alpha Vantage
            if "SSNLF" not in earnings_results:
                earnings_results["SSNLF"] = {
                    "reported_eps": 1.96,
                    "estimated_eps": 2.00,
                    "surprise_percentage": -2.0
                }
                
            return earnings_results
        except Exception as e:
            logger.error(f"Error fetching earnings data: {str(e)}")
            return {}

# Scraping Agent for news and filings
class ScrapingAgent(Agent):
    def __init__(self):
        super().__init__("Scraping Agent")
        
    async def process(self, input_data):
        logger.info(f"{self.name} processing: {input_data}")
        
        # Get news using Alpha Vantage News API
        try:
            news_data = await self.get_news_data("asia tech")
            return {"news": news_data}
        except Exception as e:
            logger.error(f"Error in scraping agent: {str(e)}")
            return {"error": "Could not fetch news data"}
    
    async def get_news_data(self, topics):
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={topics}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            if "feed" in data:
                # Extract relevant news items
                news_items = []
                for item in data["feed"][:5]:  # Limit to 5 news items
                    news_items.append({
                        "title": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "sentiment": item.get("overall_sentiment_score", 0),
                        "url": item.get("url", ""),
                        "time_published": item.get("time_published", "")
                    })
                return news_items
            return []
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return []

# Retriever Agent for RAG
class RetrieverAgent(Agent):
    def __init__(self, vector_store):
        super().__init__("Retriever Agent")
        self.vector_store = vector_store
        
    async def process(self, input_data):
        logger.info(f"{self.name} processing: {input_data}")
        
        # Generate embeddings for the query using Gemini
        query_embedding = await self.generate_embedding(input_data)
        
        # Retrieve relevant information
        if query_embedding:
            results = self.vector_store.similarity_search(query_embedding)
            return {"retrieved_data": results}
        
        return {"error": "Could not generate embeddings for retrieval"}
    
    async def generate_embedding(self, text):
        try:
            # In a production environment, you'd use a dedicated embedding model
            # For this demo, we'll create a simple embedding
            # This is a placeholder - in reality, you would use a proper embedding model
            return [0.1] * 768  # 768-dimensional embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    async def index_data(self, texts, metadata=None):
        try:
            # Generate embeddings for the texts
            embeddings = []
            for text in texts:
                embedding = await self.generate_embedding(text)
                if embedding:
                    embeddings.append(embedding)
            
            # Add to vector store
            if embeddings:
                self.vector_store.add_texts(texts, embeddings)
                return True
            return False
        except Exception as e:
            logger.error(f"Error indexing data: {str(e)}")
            return False

# Analysis Agent
class AnalysisAgent(Agent):
    def __init__(self):
        super().__init__("Analysis Agent")
        
    async def process(self, input_data):
        logger.info(f"{self.name} processing: {input_data}")
        
        try:
            # Extract portfolio and earnings data
            portfolio_data = input_data.get("portfolio", {})
            earnings_data = input_data.get("earnings", {})
            news_data = input_data.get("news", [])
            
            # Analyze portfolio allocation
            allocation_analysis = self.analyze_allocation(portfolio_data)
            
            # Analyze earnings surprises
            earnings_analysis = self.analyze_earnings(earnings_data)
            
            # Analyze market sentiment
            sentiment_analysis = self.analyze_sentiment(news_data)
            
            return {
                "allocation_analysis": allocation_analysis,
                "earnings_analysis": earnings_analysis,
                "sentiment_analysis": sentiment_analysis
            }
        except Exception as e:
            logger.error(f"Error in analysis agent: {str(e)}")
            return {"error": "Could not analyze the data"}
    
    def analyze_allocation(self, portfolio_data):
        current_allocation = portfolio_data.get("asia_tech_allocation", 0)
        previous_allocation = portfolio_data.get("previous_allocation", 0)
        change = current_allocation - previous_allocation
        
        return {
            "current": current_allocation,
            "previous": previous_allocation,
            "change": change,
            "change_percentage": (change / previous_allocation) * 100 if previous_allocation else 0
        }
    
    def analyze_earnings(self, earnings_data):
        results = []
        
        for symbol, data in earnings_data.items():
            surprise_percentage = data.get("surprise_percentage", 0)
            status = "beat" if surprise_percentage > 0 else "missed" if surprise_percentage < 0 else "met"
            
            results.append({
                "symbol": symbol,
                "status": status,
                "surprise_percentage": abs(surprise_percentage)
            })
        
        return results
    
    def analyze_sentiment(self, news_data):
        if not news_data:
            return {"overall": "neutral", "score": 0}
        
        # Calculate average sentiment
        total_sentiment = sum(item.get("sentiment", 0) for item in news_data)
        avg_sentiment = total_sentiment / len(news_data) if news_data else 0
        
        # Determine sentiment category
        if avg_sentiment > 0.2:
            category = "positive"
        elif avg_sentiment < -0.2:
            category = "negative"
        else:
            category = "neutral"
            
        # Add tilt based on additional factors
        tilt = ""
        if category == "neutral" and any("yield" in item.get("title", "").lower() or "yield" in item.get("summary", "").lower() for item in news_data):
            tilt = " with a cautionary tilt due to rising yields"
        
        return {
            "overall": category,
            "tilt": tilt,
            "score": avg_sentiment
        }

# Language Agent
class LanguageAgent(Agent):
    def __init__(self):
        super().__init__("Language Agent")
        
    async def process(self, input_data):
        logger.info(f"{self.name} processing: {input_data}")
        
        try:
            # Extract analysis results
            allocation_analysis = input_data.get("allocation_analysis", {})
            earnings_analysis = input_data.get("earnings_analysis", [])
            sentiment_analysis = input_data.get("sentiment_analysis", {})
            
            # Create prompt for Gemini
            prompt = f"""
            Create a concise market brief based on the following data:
            
            Portfolio Allocation:
            - Asia tech allocation: {allocation_analysis.get('current')}% of AUM
            - Previous allocation: {allocation_analysis.get('previous')}%
            - Change: {allocation_analysis.get('change')}%
            
            Earnings Surprises:
            {self.format_earnings(earnings_analysis)}
            
            Market Sentiment:
            - Overall: {sentiment_analysis.get('overall')}{sentiment_analysis.get('tilt', '')}
            
            Format the response as a brief spoken summary that a portfolio manager would find useful.
            Keep it concise and professional.
            """
            
            # Generate response using Gemini
            response = model.generate_content(prompt)
            
            return {"response": response.text}
        except Exception as e:
            logger.error(f"Error in language agent: {str(e)}")
            return {"response": "I'm sorry, but I couldn't generate a market brief at this time."}
    
    def format_earnings(self, earnings_analysis):
        if not earnings_analysis:
            return "- No earnings data available"
        
        formatted = []
        for item in earnings_analysis:
            formatted.append(f"- {item.get('symbol')} {item.get('status')} estimates by {item.get('surprise_percentage')}%")
        
        return "\n".join(formatted)

# Voice Agent
class VoiceAgent(Agent):
    def __init__(self):
        super().__init__("Voice Agent")
        
    async def process(self, input_data=None, audio_file=None):
        logger.info(f"{self.name} processing audio input")
        
        if audio_file:
            # Speech to text using AssemblyAI
            try:
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(audio_file)
                return {"text": transcript.text}
            except Exception as e:
                logger.error(f"Error in speech-to-text: {str(e)}")
                return {"error": "Could not transcribe audio"}
        
        if input_data and "response" in input_data:
            # Text to speech using AssemblyAI
            try:
                text = input_data["response"]
                synthesizer = aai.Speech()
                audio_url = synthesizer.synthesize(
                    text=text,
                    voice="male"  # You can choose different voices
                )
                return {"audio_url": audio_url}
            except Exception as e:
                logger.error(f"Error in text-to-speech: {str(e)}")
                return {"error": "Could not synthesize speech"}
        
        return {"error": "Invalid input for voice agent"}

# Orchestrator
class Orchestrator:
    def __init__(self):
        self.api_agent = APIAgent()
        self.scraping_agent = ScrapingAgent()
        self.retriever_agent = RetrieverAgent(vector_store)
        self.analysis_agent = AnalysisAgent()
        self.language_agent = LanguageAgent()
        self.voice_agent = VoiceAgent()
        
    async def process_query(self, query, audio_file=None):
        try:
            # If audio file is provided, transcribe it
            if audio_file:
                voice_result = await self.voice_agent.process(audio_file=audio_file)
                if "error" in voice_result:
                    return {"error": voice_result["error"]}
                query = voice_result["text"]
            
            # Process with API Agent
            api_result = await self.api_agent.process(query)
            if "error" in api_result:
                return {"error": api_result["error"]}
            
            # Process with Scraping Agent
            scraping_result = await self.scraping_agent.process(query)
            if "error" in scraping_result:
                return {"error": scraping_result["error"]}
            
            # Combine results for analysis
            combined_data = {**api_result, **scraping_result}
            
            # Process with Analysis Agent
            analysis_result = await self.analysis_agent.process(combined_data)
            if "error" in analysis_result:
                return {"error": analysis_result["error"]}
            
            # Process with Language Agent
            language_result = await self.language_agent.process(analysis_result)
            if "error" in language_result:
                return {"error": language_result["error"]}
            
            # Convert to speech if needed
            if audio_file:
                voice_output = await self.voice_agent.process(language_result)
                return {**language_result, **voice_output}
            
            return language_result
        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            return {"error": f"Orchestration failed: {str(e)}"}

# FastAPI app for microservices
app = FastAPI()

class QueryModel(BaseModel):
    query: str
    audio: Optional[bool] = False

@app.post("/process")
async def process_query(query_model: QueryModel, background_tasks: BackgroundTasks):
    orchestrator = Orchestrator()
    result = await orchestrator.process_query(query_model.query)
    return result

@app.post("/process_audio")
async def process_audio(audio_file: bytes, background_tasks: BackgroundTasks):
    orchestrator = Orchestrator()
    result = await orchestrator.process_query("", audio_file=audio_file)
    return result

# Scheduled task for morning brief
def schedule_morning_brief():
    now = datetime.datetime.now()
    target_time = datetime.datetime(now.year, now.month, now.day, 8, 0)  # 8 AM
    
    if now > target_time:
        target_time = target_time + datetime.timedelta(days=1)
    
    wait_seconds = (target_time - now).total_seconds()
    time.sleep(wait_seconds)
    
    # Run the morning brief
    query = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    orchestrator = Orchestrator()
    result = asyncio.run(orchestrator.process_query(query))
    
    # Schedule next day's brief
    schedule_morning_brief()

# Streamlit app
def run_streamlit():
    st.title("Financial Market Brief Assistant")
    
    st.markdown("""
    ## Morning Market Brief
    
    This assistant provides daily market briefs with a focus on your portfolio's risk exposure and earnings surprises.
    You can interact with it using text or voice input.
    """)
    
    # Text input
    query = st.text_input("Enter your query:", "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?")
    
    if st.button("Get Brief"):
        with st.spinner("Processing your request..."):
            # Call the API
            response = requests.post(
                "http://localhost:8000/process",
                json={"query": query}
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Brief Generated!")
                    st.write(result["response"])
                    
                    # Display audio player if available
                    if "audio_url" in result:
                        st.audio(result["audio_url"])
            else:
                st.error("Failed to process request")
    
    # Voice input
    st.markdown("### Voice Input")
    st.write("Record your question:")
    
    if st.button("Start Recording"):
        with st.spinner("Recording..."):
            # In a real app, you would implement browser-based recording
            # For this demo, we'll just wait a few seconds
            time.sleep(3)
            st.success("Recording complete!")
            
            # Simulate processing
            with st.spinner("Processing audio..."):
                time.sleep(2)
                st.write("Transcription: What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?")
                
                # Call the API (simulated)
                response_text = """
                Today, your Asia tech allocation is 22% of AUM, up from 18% yesterday.
                TSMC beat estimates by 4%, Samsung missed by 2%. Regional sentiment is
                neutral with a cautionary tilt due to rising yields.
                """
                
                st.success("Brief Generated!")
                st.write(response_text)
                
                # In a real app, you would play the audio response

# Main function to run the application
def main():
    # Start FastAPI server in a separate thread
    api_thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000))
    api_thread.daemon = True
    api_thread.start()
    
    # Schedule morning brief in a separate thread
    brief_thread = threading.Thread(target=schedule_morning_brief)
    brief_thread.daemon = True
    brief_thread.start()
    
    # Run Streamlit app
    run_streamlit()

if __name__ == "__main__":
    main()
