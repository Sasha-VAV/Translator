import torch
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from Translator import get_reviews_scorer

# Initialize FastAPI app
app = FastAPI()

# Set up templates folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
templates = Jinja2Templates(directory="templates")
reviews_scorer, reviews_tokenizer = get_reviews_scorer()
reviews_scorer.load_state_dict(torch.load("data/model.pth"))


def process_input(input_text: str) -> str:
    ans = ""
    review = reviews_tokenizer.Encode(input=input_text)
    review = torch.tensor(review, dtype=torch.long).unsqueeze(0).to(device)
    score = reviews_scorer(review)
    score = torch.argmax(score, dim=1).squeeze().item()
    ans += f"Your text is most likely: {'Positive' if score == 1 else 'Negative'} \n{'-'*50}\n"
    return ans


# Home page route
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route to handle form submission
@app.post("/process", response_class=HTMLResponse)
async def process(request: Request, input_text: str = Form(...)):
    output_text = process_input(input_text)
    return templates.TemplateResponse(
        "index.html", {"request": request, "output_text": output_text}
    )
