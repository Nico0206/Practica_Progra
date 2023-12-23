import shutil

import io
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile,Form
import pandas as pd
from typing import List

from pydantic import BaseModel as PydanticBaseModel

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class Salario(BaseModel):  #poner yo lo mio
    Wage: int
    Age: int
    Club: str
    League: str
    Position: str
    Apps: int
    Caps: int



class ListadoSalarios(BaseModel):
    salarios = List[Salario]

app = FastAPI(
    title="Servidor de datos",
    description="""Servimos datos de salarios, pero podríamos hacer muchas otras cosas, la la la.""",
    version="0.1.0",
)


@app.get("/retrieve_data/")
def retrieve_data ():
    dataset = pd.read_csv('SalaryPrediction.csv')
    # Convertimos el salario a entero
    dataset['Wage'] = dataset['Wage'].apply(lambda x: int(x.replace(',', '')))
    print(dataset.dtypes)
    dataset = dataset.fillna(0)
    dataset = dataset.to_dict(orient='records')
    listado = ListadoSalarios()
    listado.salarios = dataset
    return listado


@app.post("/add_salary/")
def add_salary(salary: Salario):
    try:
        # Cargar el conjunto de datos existente desde el CSV
        dataset = pd.read_csv('SalaryPrediction.csv')
        nuevo_salario = pd.DataFrame([salary.dict()])
        # Concatenar el nuevo salario al conjunto de datos existente
        dataset = pd.concat([dataset, nuevo_salario], ignore_index=True)
        dataset.to_csv('SalaryPrediction.csv', index=False)
        return {"mensaje": "Salario agregado correctamente", "datos": salary.dict()}
    #Gestion de errores
    except Exception as e:
        return {"error": f"Error al agregar salario: {str(e)}"}


