
#borrar y copiar mi codigo

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import plotly.express as px


@st.cache_data
def load_data(url: str):
    r = requests.get(url)
    print(r)
    if r.status_code != 200:
        return None
    mijson = r.json()
    listado = mijson['salarios']
    df = pd.DataFrame.from_records(listado)
    return df



#Ponemos un titulo para el dashboard
nombre = "Nicolás Montesinos Recio"
st.title(f'Analisis de Salarios en las diferentes Ligas Europeas - {nombre}')
st.write('El dashboard presenta un análisis completo de los salarios en las ligas europeas de fútbol. Incluye visualizaciones interactivas de salarios promedio por liga, permite comparar ligas y explorar detalles por club. También destaca la relación entre salarios y nacionalidades, ofrece un vistazo a las tendencias salariales según la edad y examina la distribución de salarios por posición en el campo. Este análisis proporciona una visión detallada de las dinámicas salariales en el fútbol europeo, facilitando a los usuarios la identificación de patrones y tendencias clave. Debemos tener en cuenta que estos datos estan ubicados al rededor de la temporada 22/23')
st.write('Antes de nada debo de explicar ciertas variables y resultados. Para cualquiera de vosotros que tenga poco conocimiento de futbol.')
st.title('Variables')
st.write(' Wage = Salario anual de un jugador ')
st.write(' Age = Edad actual (22/23)')
st.write(' Club = Club actual ')
st.write(' Nation = Nacionalidad')
st.write(' Position = Posición dentro del campo')
st.write(' Apps = Apariciones club')
st.write(' Caps = Apariciones por participación en partido internacional ')
st.title('Ligas')
st.write('Primiera Liga = Liga Portuguesa')
st.write('Ligue 1 Uber Eats = Liga Francesa')
st.write('Serie A = Liga Italiana')
st.write('Bundesliga = Liga Alemana')
st.write('La liga = La liga Española')
st.write('Premier League = La liga Inglesa')


# Intenta leer el archivo CSV
try:
    #dataset = load_data('http://fastapi:8000/retrieve_data') #ESTA ES LA QUE PONGO CUANDO LO VAYA A ENTREGAR
    dataset = load_data('http://localhost:8000/retrieve_data') #podria borrarla para cuando vaya a entregarlo
    st.title("Archivo CSV Completo que ha sido usado para este Analisis")
    st.write(dataset)
except Exception as e:
    st.error(f"Error al leer el archivo CSV: {e}")


#-----------------------------------------------------------------------
# Introducción
st.title('Análisis de Distribución de Posiciones en el Conjunto de Datos')
st.write('Antes de explorar las relaciones entre diferentes variables, es fundamental entender la distribución de posiciones en el conjunto de datos. El siguiente gráfico de pastel proporciona una visión general de la proporción de jugadores en cada posición en el campo. Esto nos ayudará a identificar la diversidad de roles que desempeñan los jugadores en las ligas europeas de fútbol.')

# Contar la cantidad de jugadores por posición
position_counts = dataset['Position'].value_counts()

# Crear un gráfico de pastel más profesional
fig_pie = px.pie(
    names=position_counts.index,
    values=position_counts.values,
    title='Distribución de Posiciones en el Campo<br><br>',
    labels={'label': 'Posición en el Campo', 'value': 'Número de Jugadores'},
    color_discrete_sequence=px.colors.qualitative.Set3,
    hole=0.3,
    width=600,
    height=400,
)

# Estilizar el diseño del gráfico
fig_pie.update_layout(
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, b=10, t=10),
)

# Mostrar el gráfico de pastel en Streamlit
st.plotly_chart(fig_pie)


#------------------------------------------------------------------------
#1)Hacemos una grafica League vs Wage (para poder hacer una de columnas)

ds = dataset[["League", "Wage"]]
dd = ds.groupby(["League"]).mean()
dd.sort_values(by=['Wage'], inplace=True, ascending=True)  # Ordenamos de menor a mayor inplace
dd.head(100)



# Ejemplo de creación del DataFrame
chart_data = pd.DataFrame({
    'Liga': dd.index,  # Utilizamos los índices de 'dd' como las ligas
    'Salario Promedio': dd['Wage']   # Utilizamos la columna 'Wage' de 'dd'
})

# Ajustamos el título general
st.title('Salario Promedio por Liga')
st.write('Los usuarios pueden ver un gráfico interactivo de barras que muestra el salario promedio ordenado de menor a mayor para distintas ligas europeas de fútbol. El título "Salario Promedio por Liga" encabeza la sección, seguido de una tabla de datos que presenta la información detallada. Los usuarios pueden explorar y comparar fácilmente los salarios promedio de las ligas utilizando el gráfico interactivo, donde las ligas se encuentran en el eje horizontal y los salarios promedio en el eje vertical. Este enfoque facilita la identificación de patrones y diferencias salariales entre las distintas ligas de interés.')

# Muestra la tabla de datos también si es necesario
st.dataframe(chart_data)

# Gráfico de barras
st.bar_chart(chart_data.set_index('Liga'), )

#------------------------------------------------------------
#Filtrar por liga (interaccion)
selected_league = st.selectbox('Selecciona una Liga', chart_data['Liga'])
league_data = dataset[dataset['League'] == selected_league]
st.write(league_data)

#---------------------------------------------------------------------
# Gráfico interactivo de comparación entre ligas
selected_leagues = st.multiselect('Selecciona Ligas para Comparar', chart_data['Liga'])
# Título específico para la sección del gráfico interactivo
st.markdown(f'### Comparación de Salarios Promedio entre Ligas: {", ".join(selected_leagues)}')
comparison_data = chart_data[chart_data['Liga'].isin(selected_leagues)]



# Gráfico de barras comparativo
st.bar_chart(comparison_data.set_index('Liga'))



#--------------------------------------------------------------------

#--------------------------------------------------------------------
#Grafico (quiero hacer el del media de salario y pais)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Gráfico de barras de la relación entre la nacionalidad y el salario promedio
avg_wages_by_nation = dataset.groupby('Nation')['Wage'].mean().reset_index()

# Ejemplo de creación del DataFrame para el gráfico
chart_data_nation = pd.DataFrame({
    'Nation': avg_wages_by_nation['Nation'],
    'Salario Promedio': avg_wages_by_nation['Wage']
})

# Ajustar el título general
st.title('Salario Promedio por Nacionalidad')

st.write('En esta sección, los usuarios encuentran un análisis interactivo de salarios promedio vinculado a distintas nacionalidades. A través de un gráfico de barras, se presenta visualmente el salario promedio para las cinco nacionalidades más destacadas. Además, se proporciona una tabla de datos para una revisión detallada. Los usuarios pueden explorar de manera interactiva seleccionando una nacionalidad específica a través de un menú desplegable, lo que filtra y muestra información detallada sobre salarios asociados a esa nacionalidad en particular. Este enfoque interactivo permite a los usuarios comparar y analizar los salarios promedio según diferentes nacionalidades de manera intuitiva.')

# Muestra la tabla de datos también si es necesario
st.dataframe(chart_data_nation)

# Gráfico de barras
st.bar_chart(chart_data_nation.set_index('Nation'))


# Filtrar por nacionalidad (interacción)
selected_nation = st.selectbox('Selecciona una Nacionalidad', chart_data_nation['Nation'])
nation_data = dataset[dataset['Nation'] == selected_nation]
st.write(nation_data)

#------------------------------------



#--------------------------------------
#Intentar un grafico o tabla que me de las medias de salario de cada club de la liga
import streamlit as st
import pandas as pd
import altair as alt

# Obtén la lista de ligas únicas
ligas_unicas = dataset['League'].unique()

# Título general de la aplicación
st.title('Análisis de Salarios en las Ligas Europeas')

st.write('Posteriorme, los usuarios pueden acceder a un análisis que presenta el salario promedio por club en una liga de fútbol seleccionada. Una tabla proporciona detalles específicos, mostrando los salarios promedio clasificados por club. Adicionalmente, se incluye un gráfico de barras interactivo, donde los clubes se encuentran en el eje horizontal y los salarios promedio en el eje vertical.')

# Widget para seleccionar la liga
selected_league = st.selectbox('Selecciona una Liga', ligas_unicas)

# Filtra los datos para la liga seleccionada
league_data = dataset[dataset['League'] == selected_league]

# Calcula el salario promedio por club
average_wage_by_club = league_data.groupby('Club')['Wage'].mean().reset_index()


# Título específico para la sección de la tabla
st.markdown(f'### Salario Promedio por Club en la Liga: {selected_league}')
# Muestra la tabla de datos
st.dataframe(average_wage_by_club)

# Gráfico de barras interactivo con Altair
chart = alt.Chart(average_wage_by_club).mark_bar().encode(
    x='Club',
    y='Wage'
).properties(
    width=600,
    height=400,
    title=f'Salario Promedio por Club en la Liga: {selected_league}'
)

# Muestra el gráfico con Altair
st.altair_chart(chart)

# Por ejemplo, agregar etiquetas al eje y
chart = alt.Chart(average_wage_by_club).mark_bar().encode(
    x='Club',
    y='Wage',
    tooltip=['Club', 'Wage']
).properties(
    width=600,
    height=400,
    title=f'Salario Promedio por Club en la Liga: {selected_league}'
)


#------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px


st.title('Media de Salarios por Edad')

st.write('Aquí, se analiza la relación entre la edad y el salario medio de los jugadores. Destacamos un gráfico interactivo de dispersión creado con Plotly Express, que visualiza la media salarial para cada edad. En el eje horizontal se encuentra la edad, y en el eje vertical, el salario medio. Además, se presenta una tabla con detalles numéricos para complementar la información visual.')
# Calculamos la media de salarios por edad
media_salarios = dataset.groupby('Age').agg({'Wage': 'mean'}).reset_index()

# Creamos un gráfico interactivo con Plotly Express
fig = px.scatter(
    media_salarios,
    x='Age',
    y='Wage',
    title='Media de Salarios por Edad',
    labels={'Age': 'Edad', 'Wage': 'Salario Medio'}
)

# Muestra el gráfico en Streamlit
st.plotly_chart(fig)

# Muestra la tabla de datos
st.dataframe(media_salarios)

st.title('Rango por edades')

min_age, max_age = st.slider('Selecciona un rango de edades', min_value=int(media_salarios['Age'].min()), max_value=int(media_salarios['Age'].max()), value=(int(media_salarios['Age'].min()), int(media_salarios['Age'].max())))

# Filtra los datos para el rango de edades seleccionado
filtered_data_age_range = media_salarios[(media_salarios['Age'] >= min_age) & (media_salarios['Age'] <= max_age)]

# Gráfico de barras comparativo
st.bar_chart(filtered_data_age_range.set_index('Age'))



#---------------------------------------------------------
# Gráfico interactivo de comparación entre edades
selected_ages = st.multiselect('Selecciona Edades para Comparar', media_salarios['Age'])
# Título específico para la sección del gráfico interactivo
st.markdown(f'### Comparación de Salarios Promedio entre Edades: {", ".join(map(str, selected_ages))}')

# Filtra los datos para las edades seleccionadas
comparison_data_age = media_salarios[media_salarios['Age'].isin(selected_ages)]

# Gráfico de barras comparativo
st.bar_chart(comparison_data_age.set_index('Age'))

show_details = st.checkbox('Mostrar detalles adicionales')

if show_details:
    st.write('Detalles adicionales:')
    st.write(comparison_data_age)



#-------------------------------------------------------
# Obtener el top N de clubes con los salarios promedio más altos en todas las ligas
# Top N Clubes que pagan más en una liga seleccionada

st.title('Clubes con los Salarios Promedio Más Altos en Todas las Ligas')

st.write('Mas allá, se destaca el análisis de los clubes con los salarios promedio más altos en todas las ligas. Utilizando un control deslizante, los usuarios pueden seleccionar el "top N" de clubes, y se muestra una tabla con los clubes correspondientes y sus salarios promedio. Este enfoque proporciona una visión rápida de los clubes con los salarios más altos en el conjunto de datos.')
top_clubs = st.slider('Selecciona el top N de clubes', 1, 20, 5)
# Obtener el top N de clubes con los salarios promedio más altos en todas las ligas
top_clubs_all_leagues = dataset.groupby('Club')['Wage'].mean().nlargest(top_clubs).reset_index()

# Título específico para la sección de la tabla
st.markdown(f'### Top {top_clubs} Clubes con los Salarios Promedio Más Altos en Todas las Ligas')
# Muestra la tabla de datos
st.table(top_clubs_all_leagues)

#-------------------------------------------------------

#import plotly.express as px
st.title('Estadísticas Descriptivas por Posición')

st.write('En este apartado, analizamos las variaciones salariales por posición en el campo. Un gráfico de caja interactivo destaca la distribución y posibles puntos atípicos. Las estadísticas clave, como media, mediana e IQR, se presentan directamente en el gráfico, y una tabla interactiva brinda una visión rápida de las tendencias salariales por posición')
# Colores personalizados para las cajas
box_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Crear un gráfico de caja interactivo mejorado
position_chart = px.box(
    dataset,
    x='Position',
    y='Wage',
    color='Position',  # Colorear cada caja según la posición
    points="all",
    title='Distribución de Salarios por Posición',
    color_discrete_sequence=box_colors,  # Utilizar colores personalizados
    labels={'Wage': 'Salario', 'Position': 'Posición'}
)

# Ajustar el diseño del gráfico
position_chart.update_traces(marker=dict(size=3), boxmean='sd', jitter=0.3)

# Añadir líneas de mediana y cuartiles a las cajas
position_chart.update_traces(boxpoints='all', jitter=0.3, line_color='black')

# Obtener estadísticas descriptivas por posición
position_stats = dataset.groupby('Position')['Wage'].describe().reset_index()

# Añadir texto con estadísticas a las cajas
for index, row in position_stats.iterrows():
    position_chart.add_annotation(
        x=row['Position'],
        y=row['75%'] + 50000,  # Ajustar la posición vertical del texto
        text=f"Promedio: {row['mean']:.2f}\nMediana: {row['50%']}\nIQR: {row['75%'] - row['25%']}",
        showarrow=False,
        font=dict(size=8)
    )

# Marcar puntos atípicos
position_chart.update_traces(marker=dict(color='red', size=6), selector=dict(marker=dict(color='red', size=6)))

# Añadir un título más descriptivo y etiquetas informativas
position_chart.update_layout(
    showlegend=False,
    title='Distribución de Salarios por Posición',
    xaxis_title='Posición en el Campo',
    yaxis_title='Salario',
    font=dict(size=12)
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(position_chart)

# Crear una tabla interactiva con estadísticas descriptivas

st.write(position_stats)

#--------------------------------------------
ds = dataset[["Age", "Wage"]]
dd = ds.groupby(["Age"]).mean()
dd.sort_values(by=['Wage'], inplace=True, ascending=True) # Ordenamos de menor a mayor inplace
print(np.quantile(dd.index, 0.75)) # Edad de la gente que más cobra

#---------------------------------------

st.title('Mapa de Calor')
st.write('Los colores indican la fuerza y dirección de la correlación, siendo tonos más claros para correlaciones más fuertes. '
         'Valores cercanos a 1 señalan una correlación positiva fuerte, sugiriendo que cuando una variable aumenta, la otra también tiende a hacerlo. ')
correlation_matrix = dataset[['Age', 'Wage', 'Apps', 'Caps']].corr()
fig = px.imshow(correlation_matrix)
st.plotly_chart(fig)

#-----------------------------------------
import matplotlib.pyplot as plt
plt.hist(dataset['Age'], bins=20)
plt.show()

#-------------------------------------------


#-------------------------------------------
st.title('Metodo Post - Agregar un nuevo Jugador')
st.write('El backend deberá tener un método post, que tenga sentido. Por ultimo, los usuarios tendran la posibilidad de agregar un nuevo jugador/salario al conjunto de datos actual.')
import streamlit as st
import requests

# Sección para agregar nuevo salario
st.header("Agregar Nuevo Jugador/Salario")

# Formulario para ingresar datos del nuevo salario
nuevo_salario = st.form(key='nuevo_salario_form')
with nuevo_salario:
    st.write("Ingrese los detalles del nuevo Jugador:")
    wage = st.number_input("Salario", min_value=0, step=1)
    age = st.number_input("Edad", min_value=0, step=1)
    club = st.text_input("Club")
    league = st.text_input("Liga")
    position = st.text_input("Posición")
    apps = st.number_input("Apariciones", min_value=0, step=1)
    caps = st.number_input("Caps", min_value=0, step=1)

    submit_button = st.form_submit_button(label='Agregar Jugador')

# Lógica para enviar el nuevo salario al servidor
if submit_button:
    nuevo_salario_data = {
        "Wage": wage,
        "Age": age,
        "Club": club,
        "League": league,
        "Position": position,
        "Apps": apps,
        "Caps": caps,
    }

    # Hacer una solicitud POST al servidor
    url = 'http://localhost:8000/add_salary/'
    response = requests.post(url, json=nuevo_salario_data)

    # Verificar la respuesta del servidor
    if response.status_code == 200:
        st.success("Salario agregado correctamente")
    else:
        st.error(f"Error al agregar salario: {response.text}")








