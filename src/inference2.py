from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    load_learner,
    open_image,
    get_transforms,
    models,
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio
import uuid
from PIL import Image
import os

model_name = 'model2.pkl'

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()


#learner = load_learner(Path('.'), 'export_sof_resnet34_4_emotions.pkl')
learner = load_learner(Path('.'), model_name)


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    named = uuid.uuid4()
    z = Image.open(BytesIO(bytes))
    z.save('g2'+'/'+str(named)+'.jpg')
    dictionary_list = (sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        ))
    
    N = len(dictionary_list)
    
    msj = """<p>"""
    
    for i in range(N):
        msj = msj + """ {c}: {s} <br>""".format(c=dictionary_list[i][0],s=str(dictionary_list[i][1]))
 
    msj = msj + """</p>"""

    msj2 = """ """
    
    for i in range(N):
        msj2 = msj2 + """<input type="radio" id={c} name="true_class" value={c}>
  <label for="{c}">{c}</label><br>""".format(c=dictionary_list[i][0])
    
    return HTMLResponse(
        "The predicted class was: " +dictionary_list[0][0]+ " with the following score: " + str(dictionary_list[0][1]) + ". All the available classes with their respective scores are described below:" + msj +   
        
        """       

<form action="/active_learning" method="post">
  <input type="hidden" id="img_uuid" name="img_uuid" value="{named}">
  <input type="hidden" id="predicted_class" name="predicted_class" value="{predicted}">
  <p>If that was not the correct class, select one of the below:</p>
  {radio_buttonS}

  <br>  
  
  <input type="submit" value="Submit">
  
</form>

    """.format(named=named,predicted=dictionary_list[0][0],radio_buttonS=msj2))

@app.route("/active_learning", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_uuid = data["img_uuid"]
    predicted_class = data["predicted_class"]
    true_class = data["true_class"]
    return rename_img_from_uuid(img_uuid,predicted_class,true_class)

def rename_img_from_uuid(img_uuid,predicted_class,true_class):
    os.rename('g2'+'/'+img_uuid+'.jpg','g2'+'/'+img_uuid+'_'+true_class+'_'+predicted_class+'.jpg')

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8018)
