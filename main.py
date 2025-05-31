# -*- coding: utf-8 -*-

"""
Created on Wed Apr 2nd 00:13:59 2025

@author: Anamika.Das

Main entry point for the marketing assistant application
"""
import json
from typing import Dict, Optional, Any
from langchain.prompts import ChatPromptTemplate
import pandas as pd
from collections import defaultdict

from config.azure_config import AzureConfig
from agents.campaign_assistant import MarketingAssistant
from agents.response_agent import ResponseRatePredictor
from workflows.message_workflow import run_marketing_message_generator
from prompts.templates import INTENT_CLASSIFICATION_SYSTEM_PROMPT

from models.data_models import (CampaignData, AutoData, AUTO_FIELDS, FIELD_GROUPS,
CAMPAIGN_FIELDS, SAMPLE_CAMPAIGNS)




def interpret_message_preferences(llm, user_response: str) -> Dict[str, list]:
    """
    Uses LLM to interpret if the user wants SMS/WhatsApp scripts,
    and in which languages (English, Arabic, or both).

    """
    system_prompt = """You are an assistant that identifies both channels and languages for marketing messages.
Return a Python dictionary with two keys: 'languages' and 'channels'.

Valid options:
- languages: ['English'], ['Arabic'], or ['English', 'Arabic']
- channels: ['SMS'], ['WhatsApp'], or ['SMS', 'WhatsApp']

Examples of valid responses:
{"languages": ["English"], "channels": ["SMS"]}
{"languages": ["Arabic", "English"], "channels": ["SMS", "WhatsApp"]}

Only respond with the Python dictionary. No explanations."""
    
    # Use plain message format, not ChatPromptTemplate
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_response}
    ]
    
    response = llm.invoke(messages)

    try:
        result = eval(response.content.strip())
        if isinstance(result, dict) and "languages" in result and "channels" in result:
            return result
    except Exception as e:
        print(f"Failed to parse LLM response: {response.content.strip()} \nError: {e}")

    # Fallback default
    return {
        "languages": ["English", "Arabic"],
        "channels": ["SMS", "WhatsApp"]
    }




def run_campaign_assistant() -> Optional[Dict[str, Any]]:
    """Main workflow for the campaign assistant"""
    config = AzureConfig()

    print("\n🌟 Marketing Campaign Assistant 🌟\n")
    print("(Type 'quit' anytime to cancel)\n")

    assistant = MarketingAssistant(config)
    user_input = input("How can I help with your marketing campaign today? ").strip()

    if user_input.lower() == "quit":
        print("Goodbye! Have a great day!")
        return None

    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFICATION_SYSTEM_PROMPT),
        ("human", user_input)
    ])

    intent_response = assistant.llm.invoke(intent_prompt.format_messages())
    intent = intent_response.content.strip().upper()

    common_campaigns = [
        "white wednesday", "black friday", "ramadan", "summer sale",
        "o2o", "eid", "new year", "back to school"
    ]
    mentioned_campaign = next((camp for camp in common_campaigns if camp in user_input.lower()), None)

    if intent == "AUTO-GENERATE" and mentioned_campaign:
        print(f"\nI'll create a complete {mentioned_campaign.title()} campaign plan for you based on best performance data.\n")
    
        # Collect required numeric inputs
        required_fields = ["campaign_budget", "campaign_weeks"]
        for field in required_fields:
            while True:
                value = getattr(assistant.campaign_auto_data, field, None)
                if value is not None and str(value).strip().isdigit():
                    break
                    
                next_q = assistant.generate_numeric_question(field, f"What is your {field.replace('_', ' ')}?")
                print(next_q)
                user_input = input("You: ").strip()
                
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    return None
                    
                # Use process_auto_input instead of process_input to update the correct data model
                assistant.process_auto_input(user_input)
        
        # Convert string values to integers
        budget = int(assistant.campaign_auto_data.campaign_budget)
        weeks = int(assistant.campaign_auto_data.campaign_weeks)
        
        # Generate campaign plan with weekly distribution
        campaign_segments_data = assistant.auto_generate_campaign_plan_revised(
            campaign_name=mentioned_campaign,
            budget=budget,
            weeks=weeks
        )
        
        # Create a DataFrame with campaign and weekly data
        rows = []
        
        # For each segment, create a row with the segment data + weekly data
        for segment_data in campaign_segments_data:
            weekly_data = segment_data["weekly_distribution"]
            segment = segment_data["customer_segment"]
            segment_type = segment.lower().split()[0]  # e.g., "Active Customer" -> "active"
            
            # Create a base row with segment information
            base_row = {
                "Customer Segment": segment,
                "Event": segment_data["event"],
                "Discount": segment_data["discount"],
                "Channel" : segment_data["Channel"],
                "Lifestage": segment_data["Lifestage"],
                "RFM": segment_data["RFM"],
                "Promo_segment": segment_data["Promo_segment"],
                "Nationality": segment_data["Nationality"],
                "Response Rate": f"{segment_data['response_rate']}%"
            }
            
            
            
            # Add weekly distribution data to the row
            for week_data in weekly_data:
                week_num = week_data["week"]
                base_row[f"Week {week_num} Budget"] = week_data["budget"]
                base_row[f"Week {week_num} Total Customers"] = week_data["total_customers"]
                
                # Add the right segment-specific count
                if segment_type in week_data:
                    base_row[f"Week {week_num} Customers"] = week_data[segment_type]
                else:
                    base_row[f"Week {week_num} Customers"] = week_data.get("customers", 0)
            
            # Add this completed row to our list
            rows.append(base_row)
        
        # Create the DataFrame from all rows
        df_auto_gen = pd.DataFrame(rows)
        
        # Ask about message generation preferences
        confirm_question = assistant.generate_question(
            "sms_whatsapp_opt_in",
            "Would you like SMS and WhatsApp scripts in English and Arabic?"
        )
        print(f"\n{confirm_question}")
        user_input = input("You: ").strip()
        
        # Handle user declining message generation
        if user_input.lower() in ["no", "n"]:
            print("Skipping message generation. Returning campaign plan only.")
            return df_auto_gen, intent
            
        # Process language preferences
        prefs = interpret_message_preferences(assistant.llm, user_input)
        languages = prefs["languages"]
        channels = prefs["channels"]
        
        print(f"Generating messages for languages: {languages}, channels: {channels}")
            
        # Generate marketing messages and add directly to existing DataFrame
        message_df = assistant.auto_generate_messages(campaign_segments_data, languages=languages)
        
        # Instead of using the default auto_generate_messages output, extract just the message columns
        message_columns = [col for col in message_df.columns if "Message" in col]
        for column in message_columns:
            df_auto_gen[column] = message_df[column].values
        
        return df_auto_gen, intent

    else:
        assistant.process_input(user_input)

        while True:
            next_question = assistant.get_next_question()
            if not next_question:
                break

            print(f"\n{next_question}")
            user_input = input("You: ").strip()

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                return None

            assistant.process_input(user_input)
            

        final_data = assistant.campaign_data.to_dict()
        print("\nFinal Campaign Data:")
        print(json.dumps(final_data, indent=2))

        return final_data, "QNA"
    
    
# =============================================================================
# INTEGRATED MAIN WORKFLOW
# =============================================================================


if __name__ == "__main__":
    # Initialize configuration
    config = AzureConfig()
    
    # Step 1: Run the campaign assistant to get a campaign plan.
    campaign_data, intent = run_campaign_assistant()
    if campaign_data is None:
        exit(0)
    
    if intent == "AUTO-GENERATE":
        print("\nGenerated Marketing Messages:")
        campaign_data.head(10)
        
    else:
        # Map campaign data fields to message generator input
        message_input = {
            "customer_segment": campaign_data.get("customer_segment", "None"),
            "language": campaign_data.get("language", "English"),
            "discount": campaign_data.get("discount", "None"),
            "coupon_code": campaign_data.get("coupon_code", "None"),
            "event": campaign_data.get("event", "None"),
            "channel": campaign_data.get("channel", "WhatsApp")
        }
        
        # For message generator, supply keys with names expected by the prompt
        customer_input_for_messages = {
            "customer_segment": message_input["customer_segment"],
            "language": message_input["language"],
            "available_discount": message_input["discount"],
            "coupon_code": message_input["coupon_code"],
            "ongoing_event": message_input["event"],
            "channel": message_input["channel"],
            "deployment_name": config.DEPLOYMENT_NAME
        }
        
        
        assistant = MarketingAssistant(config)
        budget = int(campaign_data.get("campaign_budget", 3000))
        weeks = int(campaign_data.get("campaign_weeks", 2))
        
        weekly_distribution = assistant.allocate_segment_budget(budget, weeks)
        print(weekly_distribution)
        
        # Generate marketing messages
        results = run_marketing_message_generator(config, [customer_input_for_messages])
        
        predictor = ResponseRatePredictor()
        response_rate = predictor.calculate_expected_response_rate(message_input)
        print(f"Expected response rate: {response_rate:.2f}%")
        
        # Display results
        print("\nGenerated Marketing Messages:")
        for result in results:
            print(f"Customer: {result['customer_name']}")
            for idx, msg in enumerate(result["message_options"], start=1):
                print(f"Option {idx}: {msg}")
            print("-" * 50)