
# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch, io, base64
from PIL import Image

app = FastAPI()
MODEL_PATH = 'checkpoints/model.pt'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
model=None

def load_model():
    global model
    if model is None:
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        from model import MyModel
        model_local = MyModel(**ckpt.get('model_args', {}))
        model_local.load_state_dict(ckpt['state_dict'])
        model_local.to(DEVICE); model_local.eval()
        model=model_local
    return model

class GenRequest(BaseModel):
    prompt:str=''
    n_samples:int=1
    seed:int=0

@app.post('/generate')
async def generate(req:GenRequest):
    m=load_model()
    if req.seed!=0:
        torch.manual_seed(req.seed)
    if hasattr(m,'sample'):
        out=m.sample(prompt=req.prompt, num_samples=req.n_samples)
    else:
        out=torch.randn((req.n_samples,3,256,256),device=DEVICE)
    results=[]
    for i in range(out.shape[0]):
        img=out[i].detach().cpu()
        if img.min()<-0.5: img=(img+1)/2
        img=torch.clamp(img,0,1)
        arr=(img.numpy().transpose(1,2,0)*255).astype('uint8')
        pil=Image.fromarray(arr)
        buf=io.BytesIO(); pil.save(buf,format='PNG')
        results.append(base64.b64encode(buf.getvalue()).decode())
    return JSONResponse({'images_base64':results})

@app.get('/health')
def health(): return {'status':'ok'}
