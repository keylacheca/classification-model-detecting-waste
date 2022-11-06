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
import base64 
import uuid
import os
from PIL import Image


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()


#learner = load_learner(Path('.'), 'export_sof_resnet34_4_emotions.pkl')
learner = load_learner(Path('.'), 'model.pkl')


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

@app.route("/feedback", methods=["GET"])
async def feedback(request):
    img=request.query_params["img"]#Papel_dni.jpg
    claseFB=request.query_params["fb"]
    nueva_clase= claseFB + "/"+claseFB+"_" + uuid.uuid4().hex + ".jpg"
    os.rename(img,nueva_clase)
    return HTMLResponse(
         """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8"/>
            <title>Clasificador de desechos</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
            
        </head>
        <body>
        <header>
        <h1 class="text-center"> Clasificador de desechos</h1>
        </header>
        
        <p class="text-center mr-3 ml-3"> Gracias por completar la encuesta <p> 
        <nav>
        <ul>
        
        <li> <p> <strong> Inserte imágenes de papel, plástico o vidrio: </strong></p>
        </li>
        </ul>
        </nav>
        </body>
        </html>
    """)

def predict_image_from_bytes(bytes):
    img_file = BytesIO(bytes)
    img_pil = Image.open(img_file)
    img = open_image(img_file)
    pred_class,pred_idx,outputs = learner.predict(img)
    formatted_outputs = ["{:.2f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim = 0)]]
    pred_probs = sorted(zip(learner.data.classes, map(str, formatted_outputs)),
                        key = lambda p: p[1],
                        reverse = True
                       )
    nueva_clase= pred_class + "/"+pred_class+"_" + uuid.uuid4().hex + ".jpg"
    img_pil.save(nueva_clase, format="JPEG")
    img_uri = base64.b64encode(open(nueva_clase, 'rb').read()).decode('utf-8')
    pred1="" +pred_probs[0][0] + ": " + pred_probs[0][1];
    pred2="" +pred_probs[1][0] + ": " + pred_probs[1][1];
    pred3="" +pred_probs[2][0] + ": " + pred_probs[2][1];
                                  
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8"/>
            <title>Clasificador de desechos</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">

        <style>
            .color-fondo{
                background: #55BE5A; 
            }
        </style>
        <body class="color-fondo">
        <h1 class="text-center"> Resultado </h1>
            <p class="text-center mr-3 ml-3"> La imagen insertada es: <b> %s </b> </p>
            <p class="text-center mr-3 ml-3"> Con una confiabilidad de:<br> <b> %s </b> <br>  <b> %s </b> <br>  <b> %s </b> <br> </p>
            <p class="text-center mr-3 ml-3"> <h2 class="text-center"> <i> Recicla, reutiliza, reduce e inventa… <i> </h2>  </p>
        </body>
        <div class = "d-flex justify-content-center">
            <img src="data:image/png;base64, %s" class="img-fluid mb-3" width=500px height=500px>
        </div>
        <a href="http://34.217.115.191/feedback?img=%s&fb=Papel" class="btn btn-primary">Papel</a>
        <a href="http://34.217.115.191/feedback?img=%s&fb=Plastico" class="btn btn-warning">Plastico</a>
        <a href="http://34.217.115.191/feedback?img=%s&fb=Vidrio" "btn btn-success">Vidrio</a>
        </html>
        """ %(pred_class,pred1,pred2,pred3,img_uri,nueva_clase,nueva_clase,nueva_clase))

@app.route("/")
def form(request):
    return HTMLResponse(
         """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8"/>
            <title>Clasificador de desechos</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">

            <style> body {background-image: url('https://previews.123rf.com/images/prapann/prapann1309/prapann130900205/22063674-fondo-caja-de-cart%C3%B3n-con-el-s%C3%ADmbolo-de-reciclaje.jpg')}
            </style>
        </head>
        <body>
        <header>
        <h1 class="text-center"> Clasificador de desechos</h1>
        </header>
        
        <p class="text-center mr-3 ml-3"> Este modelo se ha entrenado con im&aacutegenes de papel (periódicos, revistas, documentos, folletos, papeles de envolver, etc), plástico (sus derivados, latas de conservas) y vidrio (botellas de vidrio, trozos de espejos, cristales rotos, porcelana y cerámica.)<p> 
        <nav>
        <ul>
        
        <li> <p> <strong> Inserte imágenes de papel, plástico o vidrio: </strong></p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <strong> Seleccione la imagen a cargar:</strong><br> <p>
            <input type="file" name="file" class="btn btn-info"><br><p>
            <input type="submit" value="Cargar imagen" class="btn btn-success">
        </form>
        </li>
        <li>
        <strong> Subir una URL </strong>
        <form action="/classify-url" method="get">
             <input type="url" name="url" size="60"><br><p>
             <input type="submit" value="Subir" class="btn btn-warning">
        </form>
        </li>
        </ul>
        </nav>
        </body>
        </html>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)