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
import numpy as np
import base64 
import uuid
from PIL import Image
import os

model_name = 'model.pkl'

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
    img_file = BytesIO(bytes)
    img_pil = Image.open(img_file)
    img = open_image(img_file)
    pred_class,pred_idx,outputs = learner.predict(img)
    pred_probs = sorted(zip(learner.data.classes, map(float, outputs)),
                        key = lambda p: p[1],
                        reverse = True
                       )
    nueva_clase= str(pred_class) + "/"+ str(pred_class)+"_" + uuid.uuid4().hex + ".jpg"
    img_pil.save(nueva_clase, format="JPEG")
    img_uri = base64.b64encode(open(nueva_clase, 'rb').read()).decode('utf-8')
    pred1="" + str(pred_probs[0][0]) + ": " + str("{:.4f}%".format(pred_probs[0][1]*100));
    pred2="" + str(pred_probs[1][0]) + ": " + str("{:.4f}%".format(pred_probs[1][1]*100));
    pred3="" +str(pred_probs[2][0]) + ": " + str("{:.4f}%".format(pred_probs[2][1]*100));
                                  
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8" name="viewport" content="width=device-width, initial-scale=1.0">
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
            
        </body>
        <div class = "d-flex justify-content-center fluid">
            <img src="data:image/png;base64, %s" class="img-fluid mb-3" width=400px>
        </div>
        <div class = "d-flex flex-column justify-content-center fluid">
        <form action="/active_learning" method="post">
            <input type="hidden" id="img_uuid" name="img_uuid" value="%s">
            <input type="hidden" id="predicted_class" name="predicted_class" value="%s">
         <p class="text-center mr-3 ml-3"> <h2 class="text-center"> <i> Recicla, reutiliza, reduce e inventa… <i> </h2>  </p>
         <p class="text-center mr-3 ml-3" > Si la predicción no es correcta, seleccione la clase correcta:</p>
      <div class = "d-flex flex-row justify-content-center fluid">
      <input type="radio" name="true_class" value="Papel" id="papel_rb">
      <label for="papel_rb">Papel</label> <br>
      
      <input type="radio" name="true_class" value="Vidrio" id="vidrio_rb">
      <label for="vidrio_rb">Vidrio</label> <br>
      
      <input type="radio" name="true_class" value="Plastico" id="plastico_rb">
      <label for="plastico_rb">Plastico</label> 
      <br>  
      </div>
      <div class = "d-flex flex-row justify-content-center">
      <input type="submit" value="Enviar" class="btn btn-primary">
      </div>
      
      </div>

        <a href="/form" class = "d-flex justify-content-center" >Regresar a la página principal</a>
        
        </html>
        """ %(pred_class,pred1,pred2,pred3,img_uri,str(nueva_clase),str(pred_class)))

@app.route("/active_learning", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_uuid = data["img_uuid"]
    predicted_class = data["predicted_class"]
    true_class = data["true_class"]
    nueva_clase= true_class + "/"+ predicted_class+"_"+true_class+"_" + uuid.uuid4().hex + ".jpg"
    os.rename(img_uuid,nueva_clase)
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8" name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Clasificador de desechos</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
             <style>
            body {min-height: 100vh; background-image: url('https://www.elagoradiario.com/wp-content/uploads/2020/05/D%C3%ADa-Mundial-del-Reciclaje.jpg');background-size: cover; background-repeat: no-repeat; background-size: cover; background-attachment:fixed }
            </style>           
        </head>
        <body>
        <header>
        <h1 class="text-center">Clasificador de desechos</h1>
        </header>
        
        <p class="text-center mr-10 ml-10"> ¡Gracias por completar este formulario! <p>
        <a href="/form" class = "d-flex justify-content-center"> Regresar a la página principal</a>

        </html>
    """)

@app.route("/Clasificadordedesechos")
def form(request):
    return HTMLResponse(
         """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8" name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Clasificador de desechos</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">

            <style>
            body {min-height: 100vh; background-image: url('https://www.elagoradiario.com/wp-content/uploads/2020/05/D%C3%ADa-Mundial-del-Reciclaje.jpg');background-size: cover; background-repeat: no-repeat; background-size: cover; background-attachment:fixed }
            main {
            column-count:2;
        }
            </style>
        </head>
        <body  style="text-align: justify">
        <header>
        <div class="f-30">
        <h1 class="text-center mr-3 ml-3">Clasificador de desechos</h1>
        </header>
        </div>
        
        <main>
        
        <div class="container-fluid" >
    <di>
       <p class="text: mr-2"> Este modelo se ha entrenado con imágenes de: <br></p>
       <p class="text: mr-2"> <strong> Papel: </strong> Periódicos, revistas, documentos, folletos, papeles de envolver y recibos. <br></p>
       <p class="text: mr-2"> <strong> Plástico: </strong> Derivados del plástico y latas de conservas.<br></p>
       <p class="text: mr-2"> <strong> Vidrio: </strong>  Botellas de vidrio, cristales rotos, porcelana y cerámica.<br></p>

       <h3> Recomendaciones </h3>
       <ol>
       <p class="lead"> <li> La imagen debe enfocar el objeto en la parte central. </li> </p>
       <p class="lead"> <li> Tome en cuenta las imágenes de cada clase que debe insertar. </li> </p>
       </ol>
    </di>
         </div>
         </main>
         <br>
         
        <div class="container-fluid">
        <nav>
        <ul>
        
        <li> <p> <strong> OPCIÓN 1: Subir imágenes de papel, plástico o vidrio: </strong></p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <strong> Seleccione la imagen a cargar:</strong><br> <p>
            <input type="file" name="file" class="btn btn-info"><br><p>
            <input type="submit" value="Cargar imagen" class="btn btn-success">
        </form>
        </li>

        <li>
        <strong> OPCIÓN 2: Insertar una URL </strong>
        <form action="/classify-url" method="get">
             <input type="url" name="url" size="50"><br><p>
             <input type="submit" value="Subir" class="btn btn-warning">
        </form>
        </li>
        </ul>
        </nav>
        </div>
        </body>
        </html>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/Clasificadordedesechos")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8018)
