
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

#объект, по которому идентифицируются объекты в системе Feast
flower = Entity(
    name='flower',
    join_keys=['flower_id']
)

#объект, откуда брать данные относительно feature_repo
#timestamp нужен для выбора признаков за определенный момент времени
iris_source = FileSource(
    path = 'data/iris.parquet',
    timestamp_field = 'event_timestamp',
)

#объект, который описывает, какие признаки мы хотим получить
#с учетом срока актуальности данных
#в схеме описываются типы данных
iris_feature_view = FeatureView(
    name = 'iris_features',
    entities = [flower],
    ttl = timedelta(days=365),
    schema=[
        Field(name="sepal length (cm)", dtype=Float32),
        Field(name="sepal width (cm)", dtype=Float32),
        Field(name="petal length (cm)", dtype=Float32),
        Field(name="petal width (cm)", dtype=Float32),
        Field(name="label", dtype=Int64),
    ],
    online = False,
    source = iris_source
)
