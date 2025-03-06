from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import openai
import os
from typing import Optional

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RecipeRequest(BaseModel):
    ingredients: list[str]
    dietary_preference: str = "vegetarian"
    region: str = "North Indian"
    missing_ingredients: Optional[list[str]] = None

def generate_prompt(request: RecipeRequest) -> str:
    prompt = f"""Create a simple Indian {request.dietary_preference} recipe from {request.region} cuisine using: 
    {', '.join(request.ingredients)}. 
    Use only basic cooking tools (pan, pot, stove). 
    Include step-by-step instructions in simple English."""
    
    if request.missing_ingredients:
        prompt += f"\n\nSuggest alternatives for: {', '.join(request.missing_ingredients)}"
    
    return prompt

@app.post("/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    prompt = generate_prompt(request)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return {"recipe": response.choices[0].message.content}

# Optional Vision endpoint
@app.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    from vision_processing import analyze_image  # Implement separately
    ingredients = await analyze_image(image)
    return {"ingredients": ingredients}
