# Definición de los datos

## Origen de los datos

El dataset fue obtenido de la plataforma `Kaggle`, específicamente del repositorio público "**Pixel Art**" creado por el usuario Ebrahim Elgazar. El conjunto de datos contiene 89,000 imágenes de pixel art que representan varios personajes y objetos `Kaggle`.

**Detalles técnicos de la fuente**:

    * Plataforma: Kaggle (https://www.kaggle.com)
    * Identificador del dataset: ebrahimelgazar/pixel-art
    * URL de acceso: https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art/data
    * Autor/Propietario: Ebrahim Elgazar
    * Fecha de publicación: Febrero de 2024
    * Cantidad de imagenes: $89,000$
    * Tipo de contenido: Imágenes de pixel art en formato JPEG
    * Liciencia de uso: Apache 2.0, la cual permite procesar la información y sus derivados con total libertad.

**Composición del dataset**:

La obtención de los datos para su posterior manejo se realizó utilizando el script ubicado en `.\scripts\data_acquisition\main.py`, el cual descarga los siguientes componentes:

Carpeta `images/`: Contiene las $89,000$ imágenes en formato **JPEG**.
Archivo `labels.csv`: Etiquetas correspondientes a cada imagen.
Archivo `sprites.npy`: Representación vectorial (array NumPy) de las imágenes.
Archivo `sprites_labels.npy`: Representación vectorial de las etiquetas de las imágenes.

La fuente original no proporciona documentación sobre el significado de cada categoría en las etiquetas, lo que implica una limitación inicial para la interpretación directa de las clases definidas en los archivos. Esta situación requiere un análisis exploratorio posterior para mapear las etiquetas numéricas a categorías semánticas.

## Scripts para la carga de datos

Como se mencionó en la sección anterior, se utilizó el script `main.py` ubicado en `.\scripts\data_acquisition` para automatizar la fase de adquisición de datos. Este script implementa un pipeline programático que:

    1. **Descarga desde Kaggle**: Utiliza la biblioteca `kagglehub` para conectarse con la API de Kaggle y descargar el dataset identificado como `ebrahimelgazar/pixel-art` a la caché local del sistema (típicamente en `~/.cache/kagglehub/...`)
    2. **Gestión validada de rutas de destino**: Solicita interactivamente al usuario la ruta donde se almacenarán los datos, convierte la entrada a ruta absoluta mediante `os.path.abspath()`, y valida la existencia del directorio con creación automática mediante `os.makedirs()` si es necesario.
    3. **Transferencia automática desde caché temporal a ubicación definitiva**: Copia recursivamente toda la estructura de archivos desde la caché de `kagglehub` al directorio de destino especificado usando `shutil.copytree()` con el parámetro `dirs_exist_ok=True` para permitir fusión de contenido.
    4. **Verificación post-descarga**: Lista los primeros 5 archivos del directorio destino como confirmación visual de la operación exitosa.

Esta implementación garantiza reproducibilidad (cualquier miembro del equipo puede ejecutar el mismo script y obtener idénticos resultados), elimina intervención manual (no requiere descargas por navegador web), y prepara la estructura de archivos para las fases posteriores de análisis exploratorio, preprocesamiento y modelado.

## Referencias a rutas o bases de datos origen y destino

### Rutas de origen de datos

**Ubicación de los archivos de origen**:

Los datos se obtuvieron del repositorio público de Kaggle identificado como `ebrahimelgazar/pixel-art`. El proceso de descarga sigue la siguiente ruta de transferencia:

    1. **Origen remoto**: Servidores de Kaggle (`https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art/data`)
    2. **Caché temporal**: Directorio local de kagglehub (ruta típica: `~/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/X/`)
    3. **Destino definitivo**: Ruta especificada interactivamente por el usuario durante la ejecución del script `main.py`.

**Estructura de los archivos de origen**:

El dataset descargado presenta la siguiente organización jerárquica:

``` python
pixel-art/
├── images/
│   ├── image_0001.JPEG
│   ├── image_0002.JPEG
│   ├── image_0003.JPEG
│   └── ... (89,000 archivos JPEG)
├── labels.csv
├── sprites.npy
└── sprites_labels.npy
```

### Base de datos de destino

La base de datos de destino corresponde a un sistema de almacenamiento basado en archivos en el sistema local, estructurado específicamente para workflows de aprendizaje automático con imágenes.

**Tipo de almacenamiento**:

    * Sistema: Sistema de archivos local (**File system local**) en Windows
    * Paradigma: Almacenamiento orientado a objetos (cada imagen como archivo independiente)
    * Ubicación: Directorio raíz configurado durante la ejecución de main.py
    * Justificación técnica: Para datasets de imágenes, el almacenamiento en file system es más eficiente que bases de datos relacionales o NoSQL, ya que:
        * Permite acceso directo mediante I/O del sistema operativo.
        * Facilita integración con frameworks de Deep Learning (PyTorch DataLoader, TensorFlow tf.data)
        * Reduce overhead de serialización/deserialización
        * Simplifica versionado y respaldo mediante herramientas estándar

**Carga de los datos**

**Especificación de la base de datos de destino**:

La base de datos de destino corresponde a un sistema de almacenamiento basado en archivos (file-based storage) en el sistema local, estructurado específicamente para workflows de aprendizaje automático con imágenes.

**Tipo de almacenamiento**:

    * **Sistema**: File system local (NTFS en Windows, ext4/APFS en Linux/macOS)
    * **Paradigma**: Almacenamiento orientado a objetos (cada imagen como archivo JPEG independiente)
    * **Ubicación**: Directorio raíz configurado dinámicamente durante la ejecución de ``main.py``.
    * **Justificación técnica**: Para datasets de imágenes, el almacenamiento en file system es más eficiente que bases de datos relacionales (PostgreSQL, MySQL) o NoSQL (MongoDB) debido a que:

        * Permite acceso directo mediante I/O del sistema operativo sin overhead de queries SQL
        * Facilita integración nativa con frameworks de Deep Learning (PyTorch `DataLoader`, TensorFlow `image_dataset_from_directory`)
        * Reduce latencia al eliminar serialización/deserialización de BLOBs
        * Simplifica versionado y respaldo mediante herramientas estándar del sistema operativo (`rsync`, `git-lfs`)

**Estructura de la base de datos de destino**:

La organización actual del destino replica la estructura original del repositorio de Kaggle:

``` python
[ruta_especificada_por_usuario]/
├── images/
│   ├── image_0001.JPEG
│   ├── image_0002.JPEG
│   └── ... (89,000 archivos)
├── labels.csv
├── sprites.npy
└── sprites_labels.npy
```
**Estructura proyectada para fases de procesamiento**:

```python
proyecto_pixel_art/
├── data/
│   ├── raw/                          # Datos originales (destino de main.py)
│   │   ├── images/
│   │   ├── labels.csv
│   │   ├── sprites.npy
│   │   └── sprites_labels.npy
│   │
│   ├── processed/                    # Datos transformados (etapas futuras)
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   │
│   └── metadata/                     # Archivos auxiliares de análisis
│       ├── dataset_info.csv
│       ├── label_mapping.json
│       └── train_val_test_splits.txt
│
└── scripts/
    └── data_acquisition/
        └── main.py
```

**Procedimientos de carga y transformación**

**Adquisición automática (implementada en main.py)**:

El script ejecuta el siguiente flujo de operaciones:

    1. **Conexión con API de Kaggle**: Mediante kagglehub.dataset_download(`ebrahimelgazar/pixel-art`) establece autenticación con las credenciales del sistema (archivo `~/.kaggle/kaggle.json`)
    2. **Descarga a caché temporal**: Los archivos se descargan a `~/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/X/` donde `X` es la versión del dataset.

    3. **Configuración de destino definitivo**:

    ``` python
    python   path_input = input("Ingresa la ruta de la carpeta donde quieres guardar las imágenes: ").strip()
    destination_path = os.path.abspath(path_input)
    ```

    El usuario especifica la ruta destino, que se normaliza a ruta absoluta para evitar ambigüedades

    4. **Validación y creación de estructura**:

    ```python
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    ```

    Verifica existencia del directorio y crea toda la jerarquía necesaria si no existe, con manejo de excepciones OSError para permisos insuficientes o rutas inválidas

    5. **Transferencia de archivos**:

    ``` python
    shutil.copytree(path_descarga, destination_path, dirs_exist_ok=True)
    ```

    Copia recursiva de toda la estructura preservando metadatos (timestamps, permisos). El parámetro dirs_exist_ok=True permite fusionar contenido si el destino ya contiene archivos

    6. **Verificación post-carga**:

    ```python
    print("Archivos en tu carpeta de proyecto:", os.listdir(destination_path)[:5])
    ```
    Lista los primeros 5 elementos como confirmación visual de éxito

    **Características del proceso de carga**:

        * **Tiempo estimado**: 5-15 minutos (depende de velocidad de conexión y I/O de disco)
        * **Espacio requerido**: Aproximadamente 750 MB (89,000 imágenes JPEG + archivos NumPy)
        * **Reproducibilidad**: Ejecución idempotente - múltiples ejecuciones no causan duplicación gracias a `dirs_exist_ok=True`.
        * **Preservación de origen**: Uso de `copytree` en lugar de `move` mantiene archivos en caché de kagglehub para reutilización en otros proyectos.

**Procesamiento del archivo labels.csv**

El archivo `labels.csv` tiene las etiquetas de todas las imagenes, pero los nombres de los archivos respecto a los del .csv tiene un desface en la enumeración, ya que en el `.csv` se inicia con `image_1.jpg` y termina con `image_89400.jpg`; pero en los nombres de las imagenes se `image_0.JPEG` y termina con `image_89399.JPEG`. Por este motivo, se hace el cambio de los nombres dentro del archivo `.csv`, para que coincidan con los nombres de las imagenes tanto en los números como en el formato. De esta manera podemos garantizar la posibilidad de validar mas adelante, que cada una de las imagenes tenga una etiqueta. Para garantizar la coherencia entre archivos y etiquetas se utiliza el script `correcion_nombres_imagenes.py` ubicado en `.\scripts\data_acquisition`.
