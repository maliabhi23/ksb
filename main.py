from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import time
import pandas as pd

# Import the backend analysis code
from backend_analysis import (
    load_dataset,
    train_model,
    calculate_predicted_demand,
    distribute_energy,
    generate_prediction_map,
    generate_distribution_map
)

app = FastAPI()

# Set up the templates directory and static files directory
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the dataset and train the model once during app startup
df = load_dataset()
model = train_model(df)

# Global variable to store the last result file path
last_result_file = None

# Route to serve the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for energy prediction
@app.post("/predict")
async def predict(request: Request):
    global last_result_file  # Track the file path globally

    body = await request.json()
    date = body.get("date")
    energy = body.get("energy")

    predicted_demand_df = calculate_predicted_demand(df, date, model)
    heatmap_path = generate_prediction_map(predicted_demand_df)

    # Save the predicted demand to an Excel file
    last_result_file = os.path.join("static", f"predicted_demand_{int(time.time())}.xlsx")
    predicted_demand_df.to_excel(last_result_file, index=False)

    # Append a timestamp to the URL to prevent caching
    heatmap_url = f"/static/{os.path.basename(heatmap_path)}?t={int(time.time())}"
    return JSONResponse(content={"heatmap_url": heatmap_url})

# Route for energy distribution
@app.post("/distribute")
async def distribute(request: Request):
    global last_result_file  # Track the file path globally

    body = await request.json()
    date = body.get("date")
    energy = body.get("energy")

    # Calculate predicted demand for the given date using the trained model
    predicted_demand_df = calculate_predicted_demand(df, date, model)

    # Distribute energy based on the predicted demand
    allocation_result_df = distribute_energy(energy, predicted_demand_df)

    # Save the distribution results to an Excel file
    last_result_file = os.path.join("static", f"energy_distribution_{int(time.time())}.xlsx")
    allocation_result_df.to_excel(last_result_file, index=False)

    # Generate the distribution heatmap
    heatmap_path = generate_distribution_map(allocation_result_df)

    # Return the heatmap URL to the frontend
    return JSONResponse(content={"heatmap_url": f"/static/{os.path.basename(heatmap_path)}?t={int(time.time())}"})

# Route to download the last result file
@app.get("/download/{action}")
async def download(action: str):
    if last_result_file and os.path.exists(last_result_file):
        return FileResponse(last_result_file, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=os.path.basename(last_result_file))
    return JSONResponse(content={"error": "No file available for download"}, status_code=404)
