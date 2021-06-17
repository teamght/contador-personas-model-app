FROM python:3.8

WORKDIR /contador-personas-model-app

COPY ["requirements.txt", "."]

COPY ./mobilenet_ssd ./mobilenet_ssd

COPY ["./pyimagesearch", "./pyimagesearch"]

COPY ["./templates", "./templates"]

COPY ["people_counter.py", "."]

EXPOSE 5000

RUN apt-get update && apt-get install -y build-essential cmake python3-opencv

ENV PYTHONPATH="/contador-personas-model-app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

RUN ["pip", "install", "-r", "requirements.txt"]

CMD ["python", "people_counter.py"]