FROM python:3.8

WORKDIR /tessting

COPY ["requirements.txt", "."]

COPY ./mobilenet_ssd ./mobilenet_ssd

COPY ["./output", "./output"]

COPY ["./pyimagesearch", "./pyimagesearch"]

COPY ["./videos", "./videos"]

COPY ["./templates", "./templates"]

COPY ["people_counter.py", "."]

EXPOSE 5000

RUN apt-get update && apt-get install -y build-essential cmake python3-opencv

ENV PYTHONPATH="/tessting:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

RUN ["pip", "install", "-r", "requirements.txt"]

CMD python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/webcam_output.avi