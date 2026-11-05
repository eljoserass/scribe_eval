# scribe eval

un evaluation harness para escribas digitales medicos.

éstos dispositivos uselen encenderse durante la consulta para producir una nota médica.

las notas medicas por lo general siguen el formato SOAP(Subjective, Objective, Assesment, Plan), producidas principalmente con información disponible en la conversación.

el dispositivo escucha a la conversación entre el paciente y el personal médico y produce una nota que el medico verifica e introduce al sistema de salud.

su funcionamiento en principio es bastante similar a los generadores de resúmenes de reuniones, diferenciandose principalmente:
- la privacidad de datos es crurcial, el sistema estará en contanto posiblemente con historiales médicos, y con conversaciones privadas del paciente. cada hospital tiene su política de datos respectiva. local es bastante importante.
- errores de transcripción pueden tener repercuciones graves. apesar de que esto cuenta como soporte documental, el impacto en que alguien diga que tiene o NO tiene un sintoma es mas importante que los errores que podria tener un resumidor de notas general.
- la integracion es importante, se tiene que integrar con infraestructura acutal de IT y EHS, con dejar el texto del resultado en un bloque de texto no basta.

overall, las cosas que tenemos que evaluar para garantizar su funcionamiento seguro son
- captura de audio: que pueda procesar el audio en un principio y que tipo de audio no puede procesar
- transcripcion: pasar de audio a texto
- diarizacion: saber quien dijo que o cuando. tambien evaluar el efecto realmente de la diarizacion que no necesariamente puede ser beneficioso, o incluso una diarizacion mala quizas es mas perjudicial que transcripcion sin diarizacion.
- nota: se puede evaluar sin nota de referencia, teniendo en cuenta si usa solo informacion presente en la transcripcion, que no agregue cosas que no deberían. tambien se puede usar nota de referencia.


para probar cada uno de los bordes de funcionalidad del dispositivo en conjunto hay que verificar cada parte, cada una tiene diferentes datasets, con diferentes metricas y tipos de perturbacion (eg, cambiar un pedazo en el audio o agregar ruido, vs cambiar el orden de los speakers para ver como afecta la nota) 

hay datasets que son especificos para cada parte, algunos que funcionan para varias partes de la pipeline, algunos medicos, algunos en otros idiomas.

tambien para evaluar la pipeline hay que tomar en cuenta uso de recursos, eg token usados, tiempo que tarda, ram, operaciones, etc. basicamente guardar y tomar en cuenta con una metrica los datos de telemetry que reporte el escriba.

cosas de integracion y operacionales de como funciona para medico ya son aparte. se podria considerar quizas una feature para obtener los datos y ejecutar el analisis

at minimum necesita una categorizar los steps de la pipeline que son posible evaluar. 

maybe se podria considerar artefactos y metricas evaluables. por ejemplo una nota es un artefacto, por ahora la forma basica es texto, y otra forma trivial de evaluarlo seria por ejemplo rouge, aunque rouge luego especifica que necesita un artefacto de referencia para comparar. 

las notas pueden tener muchas metricas asi, y que las metricas tengan sus requirements, por ejemplo otro puede ser que información excluida/añadida. puede pedir una transcripcion y computar alguna evaluacion de que por ejemplo se menciona paracetamol en la transcripcion y la menciona en el texto, luego lo contrario, incluye paracetamol pero no se menciona nunca. aqui quizas haya que tener en cuenta metodos de extraccion de entidades. 

luego ver como combate sobre perturbations podria ser, a la hora de generar la nota, evaluas la nota limpia, luego evaluas cuando perturbas la transcripcion y ves la diferencia.

se definiria basicamente artefactos como
audio
transcripcion
diarizacion
transcripcion diarizada
nota


puede haber pipelines que vayan directamente de audio a nota por lo que solo se pueda evaluar audio y nota

otros tendran todas las partes, y definen que inputs requieren.

muchas de estas metricas seran estandars

es mas que el harness sea el glue entre, data, metricas, y pipelines

por ahora esta opinionated de que funcione con la arquitectura de scribnic que es el que vamos a evaluar principalmente.

algunas referencias 


https://huggingface.co/collections/sarahgebauer/ambient-scribe-evaluation-datasets

https://aclanthology.org/2025.naacl-long.426.pdf

https://arxiv.org/html/2508.01401v2

https://arxiv.org/html/2408.14568v1

GOOD RESOURCE ON SOAP https://www.ncbi.nlm.nih.gov/books/NBK482263/

https://www.nature.com/articles/s41746-025-01622-1

https://github.com/AutoEvaluation/ADS_evaluation

https://github.com/AutoEvaluation/EvalLLM4Health



 
